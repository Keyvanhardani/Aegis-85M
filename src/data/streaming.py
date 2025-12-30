# EFC Streaming Data Module
# Memory-efficient data loading for large-scale training
# Supports HuggingFace datasets, local files, and streaming sources
# ASCII-only source code to prevent encoding issues

import os
import random
from typing import (
    Dict,
    List,
    Optional,
    Iterator,
    Tuple,
    Union,
    Callable,
    Any,
    Protocol,
    runtime_checkable,
)
from pathlib import Path
from dataclasses import dataclass, field
import json
import gzip
import hashlib

import torch
from torch.utils.data import IterableDataset, DataLoader


@runtime_checkable
class TokenizerProtocol(Protocol):
    """Protocol for tokenizer interface."""

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        """Encode text to token IDs."""
        ...

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        ...

    @property
    def eos_id(self) -> int:
        """Return EOS token ID."""
        ...

    @property
    def pad_id(self) -> int:
        """Return PAD token ID."""
        ...


@dataclass
class StreamingConfig:
    """
    Configuration for streaming data loader.

    Attributes:
        seq_len: Sequence length for training samples
        buffer_size: Number of documents to buffer for shuffling
        pack_sequences: Whether to pack multiple documents into sequences
        add_eos: Add EOS token between documents when packing
        seed: Random seed for shuffling
        max_tokens: Maximum total tokens to process (None = unlimited)
        skip_tokens: Number of tokens to skip from start (for resuming)
    """

    seq_len: int = 2048
    buffer_size: int = 10000
    pack_sequences: bool = True
    add_eos: bool = True
    seed: int = 42
    max_tokens: Optional[int] = None
    skip_tokens: int = 0


class DocumentBuffer:
    """
    Shuffle buffer for streaming documents.

    Maintains a fixed-size buffer of documents and yields randomly
    sampled documents when buffer is full. Provides randomization
    without loading entire dataset.

    Args:
        buffer_size: Maximum number of documents to buffer
        seed: Random seed for sampling
    """

    def __init__(self, buffer_size: int = 10000, seed: int = 42):
        self.buffer_size = buffer_size
        self.buffer: List[str] = []
        self.rng = random.Random(seed)

    def add(self, document: str) -> Optional[str]:
        """
        Add document to buffer. Returns a random document if buffer full.

        Args:
            document: Document text to add

        Returns:
            Random document from buffer if full, None otherwise
        """
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(document)
            return None
        else:
            # Buffer full - sample random document and replace with new one
            idx = self.rng.randint(0, self.buffer_size - 1)
            result = self.buffer[idx]
            self.buffer[idx] = document
            return result

    def flush(self) -> Iterator[str]:
        """
        Yield all remaining documents in shuffled order.

        Returns:
            Iterator over remaining documents
        """
        self.rng.shuffle(self.buffer)
        for doc in self.buffer:
            yield doc
        self.buffer = []

    def __len__(self) -> int:
        return len(self.buffer)


class SequencePacker:
    """
    Packs tokenized documents into fixed-length sequences.

    Concatenates documents with EOS separators and splits into
    seq_len chunks for efficient training. Maximizes GPU utilization
    by avoiding padding waste.

    Args:
        seq_len: Target sequence length
        eos_token: EOS token ID for document separation
        pad_token: PAD token ID for final sequence padding
    """

    def __init__(
        self,
        seq_len: int,
        eos_token: int,
        pad_token: int,
    ):
        self.seq_len = seq_len
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.token_buffer: List[int] = []

    def add_tokens(self, tokens: List[int], add_eos: bool = True) -> Iterator[Tuple[List[int], List[int]]]:
        """
        Add tokens to buffer and yield complete sequences.

        Args:
            tokens: Token IDs from a document
            add_eos: Whether to append EOS after tokens

        Yields:
            Tuples of (input_ids, labels) for complete sequences
        """
        # Add tokens to buffer
        self.token_buffer.extend(tokens)
        if add_eos:
            self.token_buffer.append(self.eos_token)

        # Yield complete sequences (need seq_len + 1 for input/target split)
        while len(self.token_buffer) >= self.seq_len + 1:
            chunk = self.token_buffer[: self.seq_len + 1]
            self.token_buffer = self.token_buffer[self.seq_len:]

            # Split into input and labels (shifted by 1)
            input_ids = chunk[:-1]
            labels = chunk[1:]

            yield input_ids, labels

    def flush(self) -> Optional[Tuple[List[int], List[int]]]:
        """
        Flush remaining tokens as padded final sequence.

        Returns:
            Final (input_ids, labels) tuple if buffer has content, None otherwise
        """
        if len(self.token_buffer) < 2:
            # Need at least 2 tokens for input/label pair
            self.token_buffer = []
            return None

        # Pad to seq_len + 1
        chunk = self.token_buffer[: self.seq_len + 1]
        padding_needed = (self.seq_len + 1) - len(chunk)
        chunk.extend([self.pad_token] * padding_needed)

        self.token_buffer = []

        input_ids = chunk[:-1]
        labels = chunk[1:]

        return input_ids, labels

    def reset(self) -> None:
        """Clear the token buffer."""
        self.token_buffer = []

    @property
    def pending_tokens(self) -> int:
        """Return number of tokens in buffer."""
        return len(self.token_buffer)


class StreamingDataset(IterableDataset):
    """
    Memory-efficient streaming dataset for large-scale LM training.

    Streams documents from various sources, tokenizes on-the-fly,
    packs into fixed-length sequences, and provides shuffled batches.

    Features:
    - Memory-efficient: Only buffers a fixed number of documents
    - Shuffle buffer: Provides randomization without full dataset access
    - Sequence packing: Maximizes GPU utilization by avoiding padding
    - Multi-source: Supports local files, HuggingFace datasets, iterators
    - Resumable: Can skip tokens to resume from checkpoint

    Args:
        source: Document source (iterator, file path, or HuggingFace dataset config)
        tokenizer: Tokenizer implementing TokenizerProtocol
        config: StreamingConfig with dataset parameters
    """

    def __init__(
        self,
        source: Union[Iterator[str], str, Dict[str, Any]],
        tokenizer: TokenizerProtocol,
        config: Optional[StreamingConfig] = None,
    ):
        self.source = source
        self.tokenizer = tokenizer
        self.config = config or StreamingConfig()

        # Validate tokenizer protocol
        if not isinstance(tokenizer, TokenizerProtocol):
            raise TypeError(
                f"Tokenizer must implement TokenizerProtocol. "
                f"Got {type(tokenizer).__name__}"
            )

    def _get_document_iterator(self) -> Iterator[str]:
        """
        Create document iterator from source.

        Returns:
            Iterator yielding document strings
        """
        if isinstance(self.source, str):
            # File path - read documents from file
            yield from self._read_file_documents(self.source)
        elif isinstance(self.source, dict):
            # HuggingFace dataset config
            yield from self._read_hf_documents(self.source)
        else:
            # Assume it's already an iterator
            yield from self.source

    def _read_file_documents(self, path: str) -> Iterator[str]:
        """
        Read documents from file (supports .txt, .jsonl, .jsonl.gz).

        Args:
            path: Path to document file

        Yields:
            Document strings
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Document file not found: {path}")

        # Determine file type and reader
        if path.suffix == ".gz":
            opener = gzip.open
            inner_suffix = path.stem.split(".")[-1] if "." in path.stem else "txt"
        else:
            opener = open
            inner_suffix = path.suffix[1:] if path.suffix else "txt"

        with opener(path, "rt", encoding="utf-8") as f:
            if inner_suffix == "jsonl":
                # JSONL format: one JSON object per line
                for line in f:
                    line = line.strip()
                    if line:
                        obj = json.loads(line)
                        # Common text field names
                        text = obj.get("text") or obj.get("content") or obj.get("document")
                        if text:
                            yield text
            else:
                # Plain text: treat entire file as one document or split by blank lines
                current_doc = []
                for line in f:
                    if line.strip():
                        current_doc.append(line)
                    elif current_doc:
                        yield "".join(current_doc)
                        current_doc = []
                if current_doc:
                    yield "".join(current_doc)

    def _read_hf_documents(self, config: Dict[str, Any]) -> Iterator[str]:
        """
        Read documents from HuggingFace datasets.

        Config dict should contain:
        - path: Dataset path (e.g., "HuggingFaceFW/fineweb-edu")
        - name: Config name (optional)
        - split: Split name (default "train")
        - text_column: Column containing text (default "text")
        - streaming: Whether to use streaming mode (default True)

        Args:
            config: HuggingFace dataset configuration

        Yields:
            Document strings
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "HuggingFace datasets library required. "
                "Install with: pip install datasets"
            )

        dataset_path = config.get("path")
        dataset_name = config.get("name")
        split = config.get("split", "train")
        text_column = config.get("text_column", "text")
        streaming = config.get("streaming", True)

        if not dataset_path:
            raise ValueError("HuggingFace config requires 'path' key")

        # Load dataset in streaming mode
        dataset = load_dataset(
            dataset_path,
            name=dataset_name,
            split=split,
            streaming=streaming,
            trust_remote_code=True,
        )

        for example in dataset:
            text = example.get(text_column)
            if text:
                yield text

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Iterate over training samples.

        Yields:
            Tuples of (input_ids, labels) tensors of shape [seq_len]
        """
        # Initialize components
        buffer = DocumentBuffer(
            buffer_size=self.config.buffer_size,
            seed=self.config.seed,
        )
        packer = SequencePacker(
            seq_len=self.config.seq_len,
            eos_token=self.tokenizer.eos_id,
            pad_token=self.tokenizer.pad_id,
        )

        total_tokens = 0
        skipped_tokens = 0

        # Process documents through buffer
        for doc in self._get_document_iterator():
            # Add to shuffle buffer
            buffered_doc = buffer.add(doc)

            if buffered_doc is not None:
                # Process document that was pushed out of buffer
                tokens = self.tokenizer.encode(
                    buffered_doc,
                    add_bos=False,
                    add_eos=False,
                )

                # Skip tokens for resuming
                if skipped_tokens < self.config.skip_tokens:
                    remaining_skip = self.config.skip_tokens - skipped_tokens
                    if len(tokens) <= remaining_skip:
                        skipped_tokens += len(tokens)
                        continue
                    else:
                        tokens = tokens[remaining_skip:]
                        skipped_tokens = self.config.skip_tokens

                # Pack and yield sequences
                for input_ids, labels in packer.add_tokens(
                    tokens, add_eos=self.config.add_eos
                ):
                    total_tokens += len(input_ids)

                    yield (
                        torch.tensor(input_ids, dtype=torch.long),
                        torch.tensor(labels, dtype=torch.long),
                    )

                    # Check token limit
                    if (
                        self.config.max_tokens is not None
                        and total_tokens >= self.config.max_tokens
                    ):
                        return

        # Flush remaining buffered documents
        for doc in buffer.flush():
            tokens = self.tokenizer.encode(doc, add_bos=False, add_eos=False)

            # Skip tokens for resuming
            if skipped_tokens < self.config.skip_tokens:
                remaining_skip = self.config.skip_tokens - skipped_tokens
                if len(tokens) <= remaining_skip:
                    skipped_tokens += len(tokens)
                    continue
                else:
                    tokens = tokens[remaining_skip:]
                    skipped_tokens = self.config.skip_tokens

            for input_ids, labels in packer.add_tokens(
                tokens, add_eos=self.config.add_eos
            ):
                total_tokens += len(input_ids)

                yield (
                    torch.tensor(input_ids, dtype=torch.long),
                    torch.tensor(labels, dtype=torch.long),
                )

                if (
                    self.config.max_tokens is not None
                    and total_tokens >= self.config.max_tokens
                ):
                    return

        # Flush packer
        final = packer.flush()
        if final is not None:
            input_ids, labels = final
            yield (
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long),
            )


class MultiSourceDataset(IterableDataset):
    """
    Combines multiple streaming sources with configurable sampling weights.

    Useful for mixing datasets (e.g., English + German) during training.

    Args:
        sources: List of (StreamingDataset, weight) tuples
        seed: Random seed for source sampling
    """

    def __init__(
        self,
        sources: List[Tuple[StreamingDataset, float]],
        seed: int = 42,
    ):
        if not sources:
            raise ValueError("At least one source required")

        self.sources = sources
        self.seed = seed

        # Normalize weights
        total_weight = sum(w for _, w in sources)
        self.weights = [w / total_weight for _, w in sources]

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Iterate over samples, sampling from sources according to weights.

        Yields:
            Tuples of (input_ids, labels) tensors
        """
        rng = random.Random(self.seed)

        # Create iterators for each source
        iterators = [iter(source) for source, _ in self.sources]
        active = [True] * len(iterators)

        while any(active):
            # Sample source according to weights
            active_indices = [i for i, a in enumerate(active) if a]
            if not active_indices:
                break

            active_weights = [self.weights[i] for i in active_indices]
            total = sum(active_weights)
            normalized = [w / total for w in active_weights]

            # Weighted random choice
            r = rng.random()
            cumsum = 0.0
            chosen_idx = active_indices[0]
            for idx, w in zip(active_indices, normalized):
                cumsum += w
                if r < cumsum:
                    chosen_idx = idx
                    break

            # Get next sample from chosen source
            try:
                sample = next(iterators[chosen_idx])
                yield sample
            except StopIteration:
                active[chosen_idx] = False


# Dataset connectors for common sources

@dataclass
class FineWebEduConfig:
    """Configuration for FineWeb-Edu dataset."""

    subset: str = "sample-10BT"  # Options: sample-10BT, sample-100BT, sample-350BT
    text_column: str = "text"
    streaming: bool = True

    def to_hf_config(self) -> Dict[str, Any]:
        """Convert to HuggingFace dataset config."""
        return {
            "path": "HuggingFaceFW/fineweb-edu",
            "name": self.subset,
            "split": "train",
            "text_column": self.text_column,
            "streaming": self.streaming,
        }


@dataclass
class OSCARConfig:
    """Configuration for OSCAR dataset (German subset)."""

    language: str = "de"
    text_column: str = "text"
    streaming: bool = True

    def to_hf_config(self) -> Dict[str, Any]:
        """Convert to HuggingFace dataset config."""
        return {
            "path": "oscar-corpus/OSCAR-2301",
            "name": f"de",
            "split": "train",
            "text_column": self.text_column,
            "streaming": self.streaming,
        }


@dataclass
class WikipediaDEConfig:
    """Configuration for German Wikipedia dump."""

    date: str = "20231101"  # Wikipedia dump date
    text_column: str = "text"
    streaming: bool = True

    def to_hf_config(self) -> Dict[str, Any]:
        """Convert to HuggingFace dataset config."""
        return {
            "path": "wikimedia/wikipedia",
            "name": f"{self.date}.de",
            "split": "train",
            "text_column": self.text_column,
            "streaming": self.streaming,
        }


def create_streaming_dataloader(
    source: Union[str, Dict[str, Any], Iterator[str], StreamingDataset],
    tokenizer: TokenizerProtocol,
    batch_size: int = 8,
    seq_len: int = 2048,
    buffer_size: int = 10000,
    num_workers: int = 0,
    seed: int = 42,
    max_tokens: Optional[int] = None,
) -> DataLoader:
    """
    Factory function to create streaming DataLoader.

    Args:
        source: Document source (path, HF config, iterator, or StreamingDataset)
        tokenizer: Tokenizer implementing TokenizerProtocol
        batch_size: Batch size
        seq_len: Sequence length
        buffer_size: Shuffle buffer size
        num_workers: Number of data loading workers
        seed: Random seed
        max_tokens: Maximum tokens to process

    Returns:
        PyTorch DataLoader for streaming data
    """
    if isinstance(source, StreamingDataset):
        dataset = source
    else:
        config = StreamingConfig(
            seq_len=seq_len,
            buffer_size=buffer_size,
            seed=seed,
            max_tokens=max_tokens,
        )
        dataset = StreamingDataset(
            source=source,
            tokenizer=tokenizer,
            config=config,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )


def create_bilingual_dataloader(
    english_source: Union[str, Dict[str, Any]],
    german_source: Union[str, Dict[str, Any]],
    tokenizer: TokenizerProtocol,
    batch_size: int = 8,
    seq_len: int = 2048,
    english_weight: float = 0.7,
    german_weight: float = 0.3,
    buffer_size: int = 10000,
    seed: int = 42,
) -> DataLoader:
    """
    Factory function to create bilingual (EN+DE) streaming DataLoader.

    Args:
        english_source: English document source
        german_source: German document source
        tokenizer: Tokenizer implementing TokenizerProtocol
        batch_size: Batch size
        seq_len: Sequence length
        english_weight: Sampling weight for English data
        german_weight: Sampling weight for German data
        buffer_size: Shuffle buffer size
        seed: Random seed

    Returns:
        PyTorch DataLoader mixing English and German data
    """
    config = StreamingConfig(
        seq_len=seq_len,
        buffer_size=buffer_size,
        seed=seed,
    )

    english_dataset = StreamingDataset(
        source=english_source,
        tokenizer=tokenizer,
        config=config,
    )

    german_dataset = StreamingDataset(
        source=german_source,
        tokenizer=tokenizer,
        config=config,
    )

    multi_source = MultiSourceDataset(
        sources=[
            (english_dataset, english_weight),
            (german_dataset, german_weight),
        ],
        seed=seed,
    )

    return DataLoader(
        multi_source,
        batch_size=batch_size,
        num_workers=0,  # Multi-source requires single worker
        pin_memory=True,
    )
