# EFC BPE Tokenizer Module
# Byte-Pair Encoding tokenizer for production use
# Byte-level approach handles German/multilingual text via UTF-8
# ASCII-only source code to prevent encoding issues

from typing import Dict, List, Optional, Tuple, Set
from collections import Counter
import json
import re


class BPETokenizer:
    """
    Byte-Pair Encoding (BPE) tokenizer for production use.

    Implements byte-level BPE (GPT-2 style) which:
    - Starts with 256 byte tokens as base vocabulary
    - Iteratively merges most frequent adjacent pairs
    - Handles any Unicode text via UTF-8 byte encoding
    - Supports bilingual English/German text natively

    Special tokens:
        - <PAD>: Padding token (index 0)
        - <UNK>: Unknown token (index 1) - rarely used with byte-level
        - <BOS>: Beginning of sequence (index 2)
        - <EOS>: End of sequence (index 3)
    """

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"

    # Number of base byte tokens (0-255)
    NUM_BYTES = 256
    # Number of special tokens
    NUM_SPECIAL = 4

    def __init__(
        self,
        merges: Optional[List[Tuple[int, int]]] = None,
        vocab_size: int = 8192,
    ):
        """
        Initialize BPE tokenizer.

        Args:
            merges: List of merge pairs (token_a, token_b) in order learned.
                   If None, tokenizer is untrained (byte-level only).
            vocab_size: Target vocabulary size (default 8192).
        """
        self.target_vocab_size = vocab_size

        # Special tokens at indices 0-3
        self.special_tokens = [
            self.PAD_TOKEN,
            self.UNK_TOKEN,
            self.BOS_TOKEN,
            self.EOS_TOKEN,
        ]

        # Cache special token IDs
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3

        # Byte tokens at indices 4-259 (256 bytes)
        # Each byte value maps to index: byte_value + NUM_SPECIAL
        self._byte_offset = self.NUM_SPECIAL

        # Merge operations: list of (token_a, token_b) pairs in merge order
        # Each merge creates a new token with index = NUM_SPECIAL + NUM_BYTES + merge_index
        self.merges: List[Tuple[int, int]] = merges if merges is not None else []

        # Build merge lookup for fast encoding
        # Maps (token_a, token_b) -> merged_token_id
        self._build_merge_lookup()

        # Build vocabulary for decoding
        self._build_vocab()

    def _build_merge_lookup(self) -> None:
        """Build merge pair to token ID lookup dictionary."""
        self.merge_lookup: Dict[Tuple[int, int], int] = {}
        merge_start = self._byte_offset + self.NUM_BYTES
        for i, (a, b) in enumerate(self.merges):
            self.merge_lookup[(a, b)] = merge_start + i

    def _build_vocab(self) -> None:
        """Build vocabulary mapping from token ID to bytes/string."""
        # Token ID -> bytes representation
        self.id_to_bytes: Dict[int, bytes] = {}

        # Special tokens (stored as empty bytes, handled specially in decode)
        for i in range(self.NUM_SPECIAL):
            self.id_to_bytes[i] = b""

        # Byte tokens: each byte value maps to itself
        for byte_val in range(self.NUM_BYTES):
            token_id = self._byte_offset + byte_val
            self.id_to_bytes[token_id] = bytes([byte_val])

        # Merged tokens: concatenation of constituent token bytes
        merge_start = self._byte_offset + self.NUM_BYTES
        for i, (a, b) in enumerate(self.merges):
            token_id = merge_start + i
            self.id_to_bytes[token_id] = self.id_to_bytes[a] + self.id_to_bytes[b]

    @property
    def vocab_size(self) -> int:
        """Return current vocabulary size."""
        return self.NUM_SPECIAL + self.NUM_BYTES + len(self.merges)

    @property
    def num_merges(self) -> int:
        """Return number of learned merge operations."""
        return len(self.merges)

    def _text_to_bytes(self, text: str) -> List[int]:
        """Convert text to list of byte values."""
        return list(text.encode("utf-8"))

    def _bytes_to_text(self, byte_list: List[int]) -> str:
        """Convert list of byte values to text."""
        return bytes(byte_list).decode("utf-8", errors="replace")

    def _get_pair_counts(
        self,
        token_sequences: List[List[int]],
    ) -> Counter:
        """
        Count frequency of adjacent token pairs across all sequences.

        Args:
            token_sequences: List of token ID lists

        Returns:
            Counter mapping (token_a, token_b) -> count
        """
        pair_counts: Counter = Counter()
        for seq in token_sequences:
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                pair_counts[pair] += 1
        return pair_counts

    def _get_pair_counts_optimized(
        self,
        word_freqs: Dict[Tuple[int, ...], int],
    ) -> Counter:
        """
        Count frequency of adjacent token pairs using word frequencies.

        Optimized version that operates on word frequency dictionary
        instead of full token sequences. Much faster for large corpora.

        Args:
            word_freqs: Dictionary mapping word tuple -> frequency count

        Returns:
            Counter mapping (token_a, token_b) -> count
        """
        pair_counts: Counter = Counter()
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_counts[pair] += freq
        return pair_counts

    def _merge_pair_optimized(
        self,
        word_freqs: Dict[Tuple[int, ...], int],
        pair: Tuple[int, int],
        new_token: int,
    ) -> Dict[Tuple[int, ...], int]:
        """
        Apply a merge operation to word frequency dictionary.

        Optimized version that operates on word frequency dictionary
        instead of full token sequences.

        Args:
            word_freqs: Dictionary mapping word tuple -> frequency count
            pair: (token_a, token_b) to merge
            new_token: Token ID for the merged result

        Returns:
            Updated word frequency dictionary with merges applied
        """
        new_word_freqs: Dict[Tuple[int, ...], int] = {}
        a, b = pair

        for word, freq in word_freqs.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            new_word_tuple = tuple(new_word)
            if new_word_tuple in new_word_freqs:
                new_word_freqs[new_word_tuple] += freq
            else:
                new_word_freqs[new_word_tuple] = freq

        return new_word_freqs

    def _merge_pair(
        self,
        token_sequences: List[List[int]],
        pair: Tuple[int, int],
        new_token: int,
    ) -> List[List[int]]:
        """
        Apply a merge operation across all sequences.

        Args:
            token_sequences: List of token ID lists
            pair: (token_a, token_b) to merge
            new_token: Token ID for the merged result

        Returns:
            Updated token sequences with merges applied
        """
        merged_sequences = []
        a, b = pair

        for seq in token_sequences:
            new_seq = []
            i = 0
            while i < len(seq):
                # Check if current position matches the pair to merge
                if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
                    new_seq.append(new_token)
                    i += 2  # Skip both tokens
                else:
                    new_seq.append(seq[i])
                    i += 1
            merged_sequences.append(new_seq)

        return merged_sequences

    def train(
        self,
        texts: List[str],
        vocab_size: Optional[int] = None,
        min_frequency: int = 2,
        verbose: bool = False,
        use_optimized: bool = True,
    ) -> None:
        """
        Train BPE tokenizer on text corpus.

        Args:
            texts: List of training texts
            vocab_size: Target vocabulary size (overrides init value)
            min_frequency: Minimum pair frequency to consider for merge
            verbose: Print training progress
            use_optimized: Use word-frequency optimization (faster for large corpora)
        """
        if vocab_size is not None:
            self.target_vocab_size = vocab_size

        # Calculate number of merges needed
        num_merges = self.target_vocab_size - self.NUM_SPECIAL - self.NUM_BYTES
        if num_merges <= 0:
            # No merges needed - pure byte-level tokenization
            self.merges = []
            self._build_merge_lookup()
            self._build_vocab()
            return

        if use_optimized:
            self._train_optimized(texts, num_merges, min_frequency, verbose)
        else:
            self._train_basic(texts, num_merges, min_frequency, verbose)

    def _train_basic(
        self,
        texts: List[str],
        num_merges: int,
        min_frequency: int,
        verbose: bool,
    ) -> None:
        """Basic training method (slower, for small corpora)."""
        # Convert all texts to byte sequences (as token IDs)
        token_sequences = []
        for text in texts:
            byte_vals = self._text_to_bytes(text)
            # Convert byte values to token IDs (offset by NUM_SPECIAL)
            token_ids = [self._byte_offset + b for b in byte_vals]
            if token_ids:  # Skip empty sequences
                token_sequences.append(token_ids)

        if not token_sequences:
            self.merges = []
            self._build_merge_lookup()
            self._build_vocab()
            return

        # Clear existing merges and rebuild from scratch
        self.merges = []
        next_token_id = self._byte_offset + self.NUM_BYTES

        # Iteratively find and apply merges
        for merge_idx in range(num_merges):
            # Count pair frequencies
            pair_counts = self._get_pair_counts(token_sequences)

            if not pair_counts:
                # No more pairs to merge
                break

            # Find most frequent pair
            best_pair, best_count = pair_counts.most_common(1)[0]

            if best_count < min_frequency:
                # No pair meets minimum frequency threshold
                break

            # Record merge and apply it
            self.merges.append(best_pair)
            token_sequences = self._merge_pair(
                token_sequences, best_pair, next_token_id
            )

            if verbose and (merge_idx + 1) % 100 == 0:
                print(
                    f"Merge {merge_idx + 1}/{num_merges}: "
                    f"({best_pair[0]}, {best_pair[1]}) -> {next_token_id} "
                    f"(freq={best_count})"
                )

            next_token_id += 1

        # Rebuild lookup tables
        self._build_merge_lookup()
        self._build_vocab()

        if verbose:
            print(f"Training complete: {len(self.merges)} merges learned")
            print(f"Final vocab size: {self.vocab_size}")

    def _train_optimized(
        self,
        texts: List[str],
        num_merges: int,
        min_frequency: int,
        verbose: bool,
    ) -> None:
        """
        Optimized training using word frequency counting.

        Instead of tracking full sequences, we maintain a dictionary of
        unique words (token tuples) and their frequencies. This is O(V*M)
        where V is vocab of unique words, vs O(N*M) for full corpus.
        """
        # Build word frequency dictionary
        # Key: tuple of token IDs, Value: count
        word_freqs: Dict[Tuple[int, ...], int] = {}

        for text in texts:
            byte_vals = self._text_to_bytes(text)
            if not byte_vals:
                continue
            # Convert byte values to token IDs
            token_ids = tuple(self._byte_offset + b for b in byte_vals)
            if token_ids in word_freqs:
                word_freqs[token_ids] += 1
            else:
                word_freqs[token_ids] = 1

        if not word_freqs:
            self.merges = []
            self._build_merge_lookup()
            self._build_vocab()
            return

        if verbose:
            total_tokens = sum(len(w) * f for w, f in word_freqs.items())
            print(f"Corpus: {len(word_freqs)} unique words, {total_tokens} total tokens")

        # Clear existing merges
        self.merges = []
        next_token_id = self._byte_offset + self.NUM_BYTES

        # Iteratively find and apply merges
        for merge_idx in range(num_merges):
            # Count pair frequencies using word freqs
            pair_counts = self._get_pair_counts_optimized(word_freqs)

            if not pair_counts:
                break

            # Find most frequent pair
            best_pair, best_count = pair_counts.most_common(1)[0]

            if best_count < min_frequency:
                break

            # Record merge and apply it
            self.merges.append(best_pair)
            word_freqs = self._merge_pair_optimized(
                word_freqs, best_pair, next_token_id
            )

            if verbose and (merge_idx + 1) % 500 == 0:
                print(
                    f"Merge {merge_idx + 1}/{num_merges}: "
                    f"({best_pair[0]}, {best_pair[1]}) -> {next_token_id} "
                    f"(freq={best_count})"
                )

            next_token_id += 1

        # Rebuild lookup tables
        self._build_merge_lookup()
        self._build_vocab()

        if verbose:
            print(f"Training complete: {len(self.merges)} merges learned")
            print(f"Final vocab size: {self.vocab_size}")

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        """
        Encode text to token IDs using learned BPE merges.

        Args:
            text: Input text string
            add_bos: Prepend BOS token
            add_eos: Append EOS token

        Returns:
            List of token IDs
        """
        if not text:
            ids = []
            if add_bos:
                ids.append(self.bos_id)
            if add_eos:
                ids.append(self.eos_id)
            return ids

        # Convert text to byte token IDs
        byte_vals = self._text_to_bytes(text)
        tokens = [self._byte_offset + b for b in byte_vals]

        # Apply merges greedily (in order learned)
        # We iterate until no more merges can be applied
        changed = True
        while changed and len(tokens) > 1:
            changed = False
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1:
                    pair = (tokens[i], tokens[i + 1])
                    if pair in self.merge_lookup:
                        new_tokens.append(self.merge_lookup[pair])
                        i += 2
                        changed = True
                        continue
                new_tokens.append(tokens[i])
                i += 1
            tokens = new_tokens

        # Add special tokens
        result = []
        if add_bos:
            result.append(self.bos_id)
        result.extend(tokens)
        if add_eos:
            result.append(self.eos_id)

        return result

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
        special_ids = {self.pad_id, self.unk_id, self.bos_id, self.eos_id}

        # Collect bytes from all tokens
        byte_list = []
        for token_id in ids:
            if skip_special_tokens and token_id in special_ids:
                continue
            if token_id in self.id_to_bytes:
                byte_list.extend(self.id_to_bytes[token_id])
            elif not skip_special_tokens:
                # Unknown token - add replacement character bytes
                byte_list.extend(b"?")

        # Convert bytes back to text
        return bytes(byte_list).decode("utf-8", errors="replace")

    def save(self, path: str) -> None:
        """
        Save tokenizer to JSON file.

        Args:
            path: Path to save tokenizer JSON
        """
        data = {
            "type": "BPETokenizer",
            "vocab_size": self.vocab_size,
            "target_vocab_size": self.target_vocab_size,
            "num_merges": len(self.merges),
            "merges": self.merges,  # List of [a, b] pairs
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """
        Load tokenizer from JSON file.

        Args:
            path: Path to tokenizer JSON file

        Returns:
            Loaded BPETokenizer instance
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Convert merge pairs from lists to tuples
        merges = [tuple(pair) for pair in data["merges"]]
        vocab_size = data.get("target_vocab_size", data.get("vocab_size", 8192))

        return cls(merges=merges, vocab_size=vocab_size)

    def get_token_string(self, token_id: int) -> str:
        """
        Get string representation of a token.

        Args:
            token_id: Token ID

        Returns:
            String representation (may contain escape sequences for bytes)
        """
        if token_id == self.pad_id:
            return self.PAD_TOKEN
        elif token_id == self.unk_id:
            return self.UNK_TOKEN
        elif token_id == self.bos_id:
            return self.BOS_TOKEN
        elif token_id == self.eos_id:
            return self.EOS_TOKEN
        elif token_id in self.id_to_bytes:
            token_bytes = self.id_to_bytes[token_id]
            try:
                return token_bytes.decode("utf-8")
            except UnicodeDecodeError:
                # Return hex representation for non-UTF8 bytes
                return "".join(f"\\x{b:02x}" for b in token_bytes)
        else:
            return f"<INVALID:{token_id}>"

    def encode_batch(
        self,
        texts: List[str],
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[List[int]]:
        """
        Encode multiple texts to token IDs.

        Args:
            texts: List of input texts
            add_bos: Prepend BOS token to each
            add_eos: Append EOS token to each

        Returns:
            List of token ID lists
        """
        return [self.encode(text, add_bos, add_eos) for text in texts]

    def decode_batch(
        self,
        id_lists: List[List[int]],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """
        Decode multiple token ID lists to texts.

        Args:
            id_lists: List of token ID lists
            skip_special_tokens: Skip special tokens in output

        Returns:
            List of decoded texts
        """
        return [self.decode(ids, skip_special_tokens) for ids in id_lists]

    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size

    def __repr__(self) -> str:
        return (
            f"BPETokenizer(vocab_size={self.vocab_size}, "
            f"num_merges={self.num_merges})"
        )


class RegexBPETokenizer(BPETokenizer):
    """
    BPE tokenizer with regex pre-tokenization (GPT-2/GPT-4 style).

    Applies regex pattern to split text into chunks before BPE encoding.
    This improves tokenization quality by:
    - Keeping words together (not merging across word boundaries)
    - Handling contractions properly
    - Preserving whitespace patterns
    """

    # GPT-2 style regex pattern for pre-tokenization
    # Matches: contractions, words, numbers, punctuation+spaces
    # Using standard regex compatible with Python's re module (no \p{} properties)
    GPT2_PATTERN = re.compile(
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""
    )

    # Simpler pattern that works across all Python versions
    # Handles Latin-1 extended characters for German/European languages
    SIMPLE_PATTERN = re.compile(
        r"""'(?:s|t|re|ve|m|ll|d)|"""  # Contractions
        r""" ?[A-Za-z\xc0-\xff]+|"""   # Words (including Latin-1 chars)
        r""" ?[0-9]+|"""               # Numbers
        r""" ?[^\s\w]+|"""             # Punctuation
        r"""\s+"""                      # Whitespace
    )

    def __init__(
        self,
        merges: Optional[List[Tuple[int, int]]] = None,
        vocab_size: int = 8192,
        pattern: Optional[re.Pattern] = None,
    ):
        """
        Initialize RegexBPETokenizer.

        Args:
            merges: List of merge pairs
            vocab_size: Target vocabulary size
            pattern: Regex pattern for pre-tokenization (default: SIMPLE_PATTERN)
        """
        super().__init__(merges=merges, vocab_size=vocab_size)
        self.pattern = pattern if pattern is not None else self.SIMPLE_PATTERN

    def _pre_tokenize(self, text: str) -> List[str]:
        """
        Split text into chunks using regex pattern.

        Args:
            text: Input text

        Returns:
            List of text chunks
        """
        return self.pattern.findall(text)

    def train(
        self,
        texts: List[str],
        vocab_size: Optional[int] = None,
        min_frequency: int = 2,
        verbose: bool = False,
    ) -> None:
        """
        Train BPE tokenizer with pre-tokenization.

        Args:
            texts: List of training texts
            vocab_size: Target vocabulary size
            min_frequency: Minimum pair frequency for merge
            verbose: Print training progress
        """
        # Pre-tokenize all texts into chunks
        all_chunks = []
        for text in texts:
            chunks = self._pre_tokenize(text)
            all_chunks.extend(chunks)

        # Train on pre-tokenized chunks
        super().train(
            texts=all_chunks,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            verbose=verbose,
        )

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        """
        Encode text with pre-tokenization.

        Args:
            text: Input text
            add_bos: Prepend BOS token
            add_eos: Append EOS token

        Returns:
            List of token IDs
        """
        if not text:
            ids = []
            if add_bos:
                ids.append(self.bos_id)
            if add_eos:
                ids.append(self.eos_id)
            return ids

        # Pre-tokenize into chunks
        chunks = self._pre_tokenize(text)

        # Encode each chunk separately (prevents cross-chunk merges)
        result = []
        if add_bos:
            result.append(self.bos_id)

        for chunk in chunks:
            # Use parent's encode without special tokens
            chunk_tokens = super().encode(chunk, add_bos=False, add_eos=False)
            result.extend(chunk_tokens)

        if add_eos:
            result.append(self.eos_id)

        return result

    def save(self, path: str) -> None:
        """Save tokenizer with pattern info."""
        data = {
            "type": "RegexBPETokenizer",
            "vocab_size": self.vocab_size,
            "target_vocab_size": self.target_vocab_size,
            "num_merges": len(self.merges),
            "merges": self.merges,
            "pattern": self.pattern.pattern,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "RegexBPETokenizer":
        """Load tokenizer from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        merges = [tuple(pair) for pair in data["merges"]]
        vocab_size = data.get("target_vocab_size", data.get("vocab_size", 8192))
        pattern = re.compile(data["pattern"]) if "pattern" in data else None

        return cls(merges=merges, vocab_size=vocab_size, pattern=pattern)

    def __repr__(self) -> str:
        return (
            f"RegexBPETokenizer(vocab_size={self.vocab_size}, "
            f"num_merges={self.num_merges})"
        )
