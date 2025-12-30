# EFC Dataset Module
# Dataset loaders for training pipeline

import os
import urllib.request
from typing import Optional, Tuple
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from .tokenizer import CharTokenizer


class TextDataset(Dataset):
    """
    Generic text dataset for language model training.

    Handles tokenization and chunking of text into fixed-length sequences.
    Uses sliding window with configurable stride for sequence extraction.

    Args:
        text: Raw text corpus
        tokenizer: CharTokenizer instance
        seq_len: Sequence length for training
        stride: Stride for sliding window (defaults to seq_len for non-overlapping)
    """

    def __init__(
        self,
        text: str,
        tokenizer: CharTokenizer,
        seq_len: int = 256,
        stride: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len

        # Encode entire text
        self.tokens = tokenizer.encode(text, add_bos=False, add_eos=False)

        # Pre-compute valid starting indices
        # We need seq_len + 1 tokens for each sample (input + target)
        self.valid_starts = list(range(
            0,
            len(self.tokens) - seq_len,
            self.stride
        ))

    def __len__(self) -> int:
        return len(self.valid_starts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single training sample.

        Returns:
            Tuple of (input_ids, labels) tensors of shape [seq_len]
            Labels are shifted by 1 for next-token prediction
        """
        start = self.valid_starts[idx]

        # Get seq_len + 1 tokens for input/target pair
        chunk = self.tokens[start:start + self.seq_len + 1]

        # Input: tokens[0:seq_len], Target: tokens[1:seq_len+1]
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        labels = torch.tensor(chunk[1:], dtype=torch.long)

        return input_ids, labels


class TinyShakespeare:
    """
    TinyShakespeare dataset loader.

    Downloads and caches the TinyShakespeare dataset (~1MB).
    Provides train/val splits and DataLoader creation.

    Usage:
        dataset = TinyShakespeare(data_dir="./data")
        train_loader = dataset.get_train_loader(batch_size=32)
    """

    URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    FILENAME = "tinyshakespeare.txt"

    def __init__(
        self,
        data_dir: str = "./data",
        seq_len: int = 256,
        val_split: float = 0.1,
        force_download: bool = False,
    ):
        """
        Initialize TinyShakespeare dataset.

        Args:
            data_dir: Directory to store/load data
            seq_len: Sequence length for training
            val_split: Fraction of data for validation
            force_download: Force re-download even if cached
        """
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.val_split = val_split

        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Download or load data
        self.raw_text = self._load_data(force_download)

        # Build tokenizer from text
        self.tokenizer = CharTokenizer.from_text(self.raw_text)

        # Create train/val splits
        self._create_splits()

    def _load_data(self, force_download: bool) -> str:
        """Download or load cached data."""
        filepath = self.data_dir / self.FILENAME

        if not filepath.exists() or force_download:
            print(f"[DATA] Downloading TinyShakespeare to {filepath}...")
            urllib.request.urlretrieve(self.URL, filepath)
            print(f"[DATA] Download complete. Size: {filepath.stat().st_size / 1024:.1f} KB")

        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        return text

    def _create_splits(self) -> None:
        """Create train/val text splits."""
        split_idx = int(len(self.raw_text) * (1 - self.val_split))

        train_text = self.raw_text[:split_idx]
        val_text = self.raw_text[split_idx:]

        self.train_dataset = TextDataset(
            text=train_text,
            tokenizer=self.tokenizer,
            seq_len=self.seq_len,
        )

        self.val_dataset = TextDataset(
            text=val_text,
            tokenizer=self.tokenizer,
            seq_len=self.seq_len,
        )

        print(f"[DATA] Dataset loaded:")
        print(f"       Vocab size: {self.tokenizer.vocab_size}")
        print(f"       Total chars: {len(self.raw_text):,}")
        print(f"       Train samples: {len(self.train_dataset):,}")
        print(f"       Val samples: {len(self.val_dataset):,}")

    def get_train_loader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> DataLoader:
        """Create training DataLoader."""
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )

    def get_val_loader(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> DataLoader:
        """Create validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return self.tokenizer.vocab_size

    def decode(self, ids: torch.Tensor) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(ids.tolist())

    def encode(self, text: str) -> torch.Tensor:
        """Encode text to token IDs."""
        return torch.tensor(
            self.tokenizer.encode(text),
            dtype=torch.long
        )
