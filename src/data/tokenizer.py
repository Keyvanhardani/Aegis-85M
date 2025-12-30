# EFC Tokenizer Module
# Character-level tokenizer for initial architecture verification
# ASCII-only encoding to prevent issues with German special characters

from typing import List, Optional
import json


class CharTokenizer:
    """
    Character-level tokenizer for architecture verification.

    Uses simple character-to-integer mapping. For production,
    this should be replaced with BPE/SentencePiece tokenizer.

    Special tokens:
        - <PAD>: Padding token (index 0)
        - <UNK>: Unknown character (index 1)
        - <BOS>: Beginning of sequence (index 2)
        - <EOS>: End of sequence (index 3)
    """

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"

    def __init__(self, vocab: Optional[List[str]] = None):
        """
        Initialize tokenizer.

        Args:
            vocab: Optional list of characters. If None, uses default ASCII.
        """
        self.special_tokens = [
            self.PAD_TOKEN,
            self.UNK_TOKEN,
            self.BOS_TOKEN,
            self.EOS_TOKEN,
        ]

        if vocab is None:
            # Default ASCII printable characters + newline
            vocab = self._default_vocab()

        self.vocab = self.special_tokens + vocab
        self.char_to_idx = {c: i for i, c in enumerate(self.vocab)}
        self.idx_to_char = {i: c for i, c in enumerate(self.vocab)}

        # Cache special token indices
        self.pad_id = self.char_to_idx[self.PAD_TOKEN]
        self.unk_id = self.char_to_idx[self.UNK_TOKEN]
        self.bos_id = self.char_to_idx[self.BOS_TOKEN]
        self.eos_id = self.char_to_idx[self.EOS_TOKEN]

    def _default_vocab(self) -> List[str]:
        """Generate default ASCII vocabulary."""
        # Printable ASCII characters (32-126) plus newline, tab, carriage return
        chars = [chr(i) for i in range(32, 127)]
        chars.extend(["\n", "\t", "\r"])
        return sorted(set(chars))

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        """
        Build tokenizer vocabulary from text corpus.

        Args:
            text: Training text to extract vocabulary from

        Returns:
            CharTokenizer with vocabulary from text
        """
        # Extract unique characters from text
        unique_chars = sorted(set(text))
        return cls(vocab=unique_chars)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        """
        Load tokenizer from JSON file.

        Args:
            path: Path to tokenizer JSON file

        Returns:
            Loaded CharTokenizer instance
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Extract vocab (excluding special tokens which are added in __init__)
        vocab = [c for c in data["vocab"] if c not in [
            cls.PAD_TOKEN, cls.UNK_TOKEN, cls.BOS_TOKEN, cls.EOS_TOKEN
        ]]
        return cls(vocab=vocab)

    def save(self, path: str) -> None:
        """
        Save tokenizer to JSON file.

        Args:
            path: Path to save tokenizer JSON
        """
        data = {
            "vocab": self.vocab,
            "vocab_size": len(self.vocab),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self.vocab)

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text string
            add_bos: Prepend BOS token
            add_eos: Append EOS token

        Returns:
            List of token IDs
        """
        ids = []

        if add_bos:
            ids.append(self.bos_id)

        for char in text:
            if char in self.char_to_idx:
                ids.append(self.char_to_idx[char])
            else:
                ids.append(self.unk_id)

        if add_eos:
            ids.append(self.eos_id)

        return ids

    def decode(
        self,
        ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode token IDs to text.

        Args:
            ids: List of token IDs
            skip_special_tokens: Skip PAD, UNK, BOS, EOS in output

        Returns:
            Decoded text string
        """
        chars = []
        special_ids = {self.pad_id, self.unk_id, self.bos_id, self.eos_id}

        for idx in ids:
            if skip_special_tokens and idx in special_ids:
                continue
            if idx in self.idx_to_char:
                chars.append(self.idx_to_char[idx])
            else:
                # Out of vocabulary - skip or add placeholder
                if not skip_special_tokens:
                    chars.append("?")

        return "".join(chars)

    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size

    def __repr__(self) -> str:
        return f"CharTokenizer(vocab_size={self.vocab_size})"
