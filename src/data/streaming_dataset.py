# EFC Streaming Dataset Module
# Streaming datasets for large-scale training (FineWeb-Edu, OSCAR, Wikipedia)
# Memory-efficient streaming with on-the-fly tokenization

import os
import json
import gzip
import random
from typing import Iterator, Optional, List, Tuple, Union, Callable
from pathlib import Path
from dataclasses import dataclass

import torch
from torch.utils.data import IterableDataset, DataLoader

from .bpe_tokenizer import BPETokenizer, RegexBPETokenizer


@dataclass
class DatasetShard:
    """Metadata for a dataset shard."""
    path: str
    num_samples: int
    language: str
    source: str


class StreamingTextDataset(IterableDataset):
    """
    Memory-efficient streaming dataset for large text corpora.

    Streams data from disk/files without loading entire dataset into memory.
    Supports JSONL, gzipped JSONL, and plain text formats.

    Features:
    - On-the-fly tokenization with BPE tokenizer
    - Document shuffling within buffer
    - Multi-shard support for distributed datasets
    - Resume capability via shard/offset tracking

    Args:
        shards: List of DatasetShard or file paths
        tokenizer: BPETokenizer instance
        seq_len: Sequence length for training samples
        shuffle_buffer: Size of document buffer for local shuffling
        text_key: Key for text field in JSONL (default: "text")
    """

    def __init__(
        self,
        shards: List[Union[DatasetShard, str]],
        tokenizer: Union[BPETokenizer, RegexBPETokenizer],
        seq_len: int = 512,
        shuffle_buffer: int = 10000,
        text_key: str = "text",
    ):
        self.shards = [
            s if isinstance(s, DatasetShard)
            else DatasetShard(path=s, num_samples=-1, language="en", source="unknown")
            for s in shards
        ]
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.shuffle_buffer = shuffle_buffer
        self.text_key = text_key

        # Token buffer for sequence assembly
        self._token_buffer: List[int] = []

    def _open_file(self, path: str):
        """Open file with appropriate handler (gzip or plain)."""
        if path.endswith(".gz"):
            return gzip.open(path, "rt", encoding="utf-8")
        return open(path, "r", encoding="utf-8")

    def _read_documents(self, path: str) -> Iterator[str]:
        """
        Read documents from file.

        Supports:
        - JSONL: One JSON object per line with text field
        - Plain text: Entire file as one document (split by blank lines)
        """
        with self._open_file(path) as f:
            if path.endswith(".jsonl") or path.endswith(".jsonl.gz"):
                # JSONL format
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        text = obj.get(self.text_key, "")
                        if text:
                            yield text
                    except json.JSONDecodeError:
                        continue
            else:
                # Plain text - split on double newlines (paragraphs)
                content = f.read()
                for doc in content.split("\n\n"):
                    doc = doc.strip()
                    if doc:
                        yield doc

    def _shuffle_documents(
        self,
        doc_iter: Iterator[str]
    ) -> Iterator[str]:
        """Apply local shuffling using a buffer."""
        buffer = []

        for doc in doc_iter:
            buffer.append(doc)
            if len(buffer) >= self.shuffle_buffer:
                random.shuffle(buffer)
                for item in buffer[:len(buffer) // 2]:
                    yield item
                buffer = buffer[len(buffer) // 2:]

        # Flush remaining
        random.shuffle(buffer)
        for item in buffer:
            yield item

    def _tokenize_and_chunk(
        self,
        doc_iter: Iterator[str]
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Tokenize documents and yield fixed-length chunks.

        Concatenates document tokens with EOS separator.
        Yields (input_ids, labels) pairs for next-token prediction.
        """
        for doc in doc_iter:
            # Tokenize with EOS separator
            tokens = self.tokenizer.encode(doc, add_bos=False, add_eos=True)
            self._token_buffer.extend(tokens)

            # Yield complete chunks
            while len(self._token_buffer) >= self.seq_len + 1:
                chunk = self._token_buffer[:self.seq_len + 1]
                self._token_buffer = self._token_buffer[self.seq_len:]

                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)
                yield input_ids, labels

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Iterate through all shards yielding training samples."""
        # Get worker info for distributed loading
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Distribute shards across workers
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            shards = [
                s for i, s in enumerate(self.shards)
                if i % num_workers == worker_id
            ]
        else:
            shards = self.shards

        # Shuffle shard order
        shards = list(shards)
        random.shuffle(shards)

        # Clear token buffer for new iteration
        self._token_buffer = []

        # Process each shard
        for shard in shards:
            doc_iter = self._read_documents(shard.path)
            if self.shuffle_buffer > 0:
                doc_iter = self._shuffle_documents(doc_iter)
            yield from self._tokenize_and_chunk(doc_iter)


class FineWebEdu:
    """
    FineWeb-Edu dataset loader.

    FineWeb-Edu is a high-quality educational web text dataset.
    This loader supports downloading subsets and streaming.

    Note: Requires `datasets` library for HuggingFace download.
    Falls back to manual download instructions if not available.

    Usage:
        dataset = FineWebEdu(
            data_dir="./data/fineweb-edu",
            tokenizer=tokenizer,
            subset_size="10BT"  # 10B tokens subset
        )
        loader = dataset.get_loader(batch_size=32)
    """

    # HuggingFace dataset identifiers
    HF_DATASET = "HuggingFaceFW/fineweb-edu"
    SUBSETS = {
        "sample": "sample-10BT",      # 10B tokens sample
        "10BT": "sample-10BT",
        "100BT": "sample-100BT",      # 100B tokens sample
        "350BT": "sample-350BT",      # 350B tokens sample
        "full": "default",            # Full dataset (1.3T tokens)
    }

    def __init__(
        self,
        data_dir: str = "./data/fineweb-edu",
        tokenizer: Optional[Union[BPETokenizer, RegexBPETokenizer]] = None,
        seq_len: int = 512,
        subset_size: str = "10BT",
        max_samples: Optional[int] = None,
    ):
        """
        Initialize FineWeb-Edu loader.

        Args:
            data_dir: Directory to cache/store data
            tokenizer: BPE tokenizer (will create default if None)
            seq_len: Sequence length for training
            subset_size: One of "sample", "10BT", "100BT", "350BT", "full"
            max_samples: Optional limit on number of samples to use
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.seq_len = seq_len
        self.subset_size = subset_size
        self.max_samples = max_samples

        # Initialize tokenizer
        if tokenizer is None:
            self.tokenizer = RegexBPETokenizer(vocab_size=32000)
        else:
            self.tokenizer = tokenizer

        # Check for cached JSONL files
        self.shard_paths = self._find_shards()

        if not self.shard_paths:
            print("[FineWeb-Edu] No cached data found.")
            print("[FineWeb-Edu] Run `python -m src.data.download_fineweb` to download.")

    def _find_shards(self) -> List[str]:
        """Find existing shard files."""
        shards = []
        for ext in ["*.jsonl", "*.jsonl.gz", "*.parquet"]:
            shards.extend(self.data_dir.glob(ext))
        return sorted([str(p) for p in shards])

    def get_streaming_dataset(
        self,
        shuffle_buffer: int = 10000,
    ) -> StreamingTextDataset:
        """
        Get streaming dataset for training.

        Args:
            shuffle_buffer: Local shuffle buffer size

        Returns:
            StreamingTextDataset instance
        """
        if not self.shard_paths:
            raise RuntimeError(
                "No data shards found. Download data first with:\n"
                "  python -m src.data.download_fineweb"
            )

        shards = [
            DatasetShard(
                path=p,
                num_samples=-1,
                language="en",
                source="fineweb-edu"
            )
            for p in self.shard_paths
        ]

        return StreamingTextDataset(
            shards=shards,
            tokenizer=self.tokenizer,
            seq_len=self.seq_len,
            shuffle_buffer=shuffle_buffer,
        )

    def get_loader(
        self,
        batch_size: int = 32,
        num_workers: int = 2,
        prefetch_factor: int = 2,
    ) -> DataLoader:
        """
        Create DataLoader for training.

        Args:
            batch_size: Training batch size
            num_workers: Number of data loading workers
            prefetch_factor: Batches to prefetch per worker

        Returns:
            DataLoader instance
        """
        dataset = self.get_streaming_dataset()

        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            pin_memory=True,
        )


class OSCARDataset:
    """
    OSCAR dataset loader for German text.

    OSCAR (Open Super-large Crawled Aggregated coRpus) provides
    web-crawled text in many languages, including German.

    Usage:
        dataset = OSCARDataset(
            data_dir="./data/oscar-de",
            tokenizer=tokenizer,
        )
        loader = dataset.get_loader(batch_size=32)
    """

    # HuggingFace dataset identifier
    HF_DATASET = "oscar-corpus/OSCAR-2301"

    def __init__(
        self,
        data_dir: str = "./data/oscar-de",
        tokenizer: Optional[Union[BPETokenizer, RegexBPETokenizer]] = None,
        seq_len: int = 512,
        language: str = "de",
    ):
        """
        Initialize OSCAR loader.

        Args:
            data_dir: Directory to cache/store data
            tokenizer: BPE tokenizer
            seq_len: Sequence length for training
            language: Language code (default: "de" for German)
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.seq_len = seq_len
        self.language = language

        # Initialize tokenizer
        if tokenizer is None:
            self.tokenizer = RegexBPETokenizer(vocab_size=32000)
        else:
            self.tokenizer = tokenizer

        # Check for cached shards
        self.shard_paths = self._find_shards()

        if not self.shard_paths:
            print(f"[OSCAR-{language.upper()}] No cached data found.")
            print(f"[OSCAR-{language.upper()}] Run download script to fetch data.")

    def _find_shards(self) -> List[str]:
        """Find existing shard files."""
        shards = []
        for ext in ["*.jsonl", "*.jsonl.gz", "*.txt"]:
            shards.extend(self.data_dir.glob(ext))
        return sorted([str(p) for p in shards])

    def get_streaming_dataset(
        self,
        shuffle_buffer: int = 10000,
    ) -> StreamingTextDataset:
        """Get streaming dataset for training."""
        if not self.shard_paths:
            raise RuntimeError(
                f"No data shards found for OSCAR-{self.language.upper()}. "
                "Download data first."
            )

        shards = [
            DatasetShard(
                path=p,
                num_samples=-1,
                language=self.language,
                source="oscar"
            )
            for p in self.shard_paths
        ]

        return StreamingTextDataset(
            shards=shards,
            tokenizer=self.tokenizer,
            seq_len=self.seq_len,
            shuffle_buffer=shuffle_buffer,
        )

    def get_loader(
        self,
        batch_size: int = 32,
        num_workers: int = 2,
    ) -> DataLoader:
        """Create DataLoader for training."""
        dataset = self.get_streaming_dataset()

        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )


class WikipediaDE:
    """
    German Wikipedia dataset loader.

    Uses preprocessed Wikipedia dumps in plain text or JSONL format.

    Usage:
        dataset = WikipediaDE(
            data_dir="./data/wikipedia-de",
            tokenizer=tokenizer,
        )
        loader = dataset.get_loader(batch_size=32)
    """

    def __init__(
        self,
        data_dir: str = "./data/wikipedia-de",
        tokenizer: Optional[Union[BPETokenizer, RegexBPETokenizer]] = None,
        seq_len: int = 512,
    ):
        """
        Initialize Wikipedia-DE loader.

        Args:
            data_dir: Directory containing preprocessed Wikipedia data
            tokenizer: BPE tokenizer
            seq_len: Sequence length for training
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.seq_len = seq_len

        # Initialize tokenizer
        if tokenizer is None:
            self.tokenizer = RegexBPETokenizer(vocab_size=32000)
        else:
            self.tokenizer = tokenizer

        # Check for cached shards
        self.shard_paths = self._find_shards()

        if not self.shard_paths:
            print("[Wikipedia-DE] No cached data found.")
            print("[Wikipedia-DE] Download Wikipedia dump and preprocess.")

    def _find_shards(self) -> List[str]:
        """Find existing shard files."""
        shards = []
        for ext in ["*.jsonl", "*.jsonl.gz", "*.txt"]:
            shards.extend(self.data_dir.glob(ext))
        return sorted([str(p) for p in shards])

    def get_streaming_dataset(
        self,
        shuffle_buffer: int = 5000,
    ) -> StreamingTextDataset:
        """Get streaming dataset for training."""
        if not self.shard_paths:
            raise RuntimeError(
                "No data shards found for Wikipedia-DE. "
                "Download and preprocess Wikipedia dump first."
            )

        shards = [
            DatasetShard(
                path=p,
                num_samples=-1,
                language="de",
                source="wikipedia"
            )
            for p in self.shard_paths
        ]

        return StreamingTextDataset(
            shards=shards,
            tokenizer=self.tokenizer,
            seq_len=self.seq_len,
            shuffle_buffer=shuffle_buffer,
        )

    def get_loader(
        self,
        batch_size: int = 32,
        num_workers: int = 2,
    ) -> DataLoader:
        """Create DataLoader for training."""
        dataset = self.get_streaming_dataset()

        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )


class BilingualMixedDataset(IterableDataset):
    """
    Mixed bilingual dataset for English/German training.

    Combines multiple data sources with configurable mixing ratios.

    Usage:
        dataset = BilingualMixedDataset(
            en_datasets=[fineweb_dataset],
            de_datasets=[oscar_de_dataset, wiki_de_dataset],
            en_ratio=0.6,  # 60% English, 40% German
        )
    """

    def __init__(
        self,
        en_datasets: List[StreamingTextDataset],
        de_datasets: List[StreamingTextDataset],
        en_ratio: float = 0.6,
    ):
        """
        Initialize mixed bilingual dataset.

        Args:
            en_datasets: List of English streaming datasets
            de_datasets: List of German streaming datasets
            en_ratio: Fraction of samples from English sources
        """
        self.en_datasets = en_datasets
        self.de_datasets = de_datasets
        self.en_ratio = en_ratio

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Iterate with mixed language sampling."""
        # Create iterators
        en_iters = [iter(ds) for ds in self.en_datasets]
        de_iters = [iter(ds) for ds in self.de_datasets]

        en_idx = 0
        de_idx = 0

        while True:
            # Sample language based on ratio
            if random.random() < self.en_ratio:
                # Sample from English
                if not en_iters:
                    continue
                try:
                    sample = next(en_iters[en_idx % len(en_iters)])
                    en_idx = (en_idx + 1) % len(en_iters)
                    yield sample
                except StopIteration:
                    # Remove exhausted iterator
                    en_iters.pop(en_idx % len(en_iters))
                    if not en_iters and not de_iters:
                        break
            else:
                # Sample from German
                if not de_iters:
                    continue
                try:
                    sample = next(de_iters[de_idx % len(de_iters)])
                    de_idx = (de_idx + 1) % len(de_iters)
                    yield sample
                except StopIteration:
                    # Remove exhausted iterator
                    de_iters.pop(de_idx % len(de_iters))
                    if not en_iters and not de_iters:
                        break
