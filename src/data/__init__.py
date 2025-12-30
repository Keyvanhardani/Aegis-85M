# EFC Data Module
# Dataset loaders and tokenizers for training

from .tokenizer import CharTokenizer
from .bpe_tokenizer import BPETokenizer, RegexBPETokenizer
from .dataset import TextDataset, TinyShakespeare
from .streaming_dataset import (
    StreamingTextDataset,
    DatasetShard,
    FineWebEdu,
    OSCARDataset,
    WikipediaDE,
    BilingualMixedDataset,
)
from .streaming import (
    StreamingDataset,
    StreamingConfig,
    MultiSourceDataset,
    DocumentBuffer,
    SequencePacker,
    FineWebEduConfig,
    OSCARConfig,
    WikipediaDEConfig,
    create_streaming_dataloader,
    create_bilingual_dataloader,
    TokenizerProtocol,
)

__all__ = [
    # Tokenizers
    "CharTokenizer",
    "BPETokenizer",
    "RegexBPETokenizer",
    # Basic datasets
    "TextDataset",
    "TinyShakespeare",
    # Streaming datasets (file-based)
    "StreamingTextDataset",
    "DatasetShard",
    "FineWebEdu",
    "OSCARDataset",
    "WikipediaDE",
    "BilingualMixedDataset",
    # Streaming datasets (HuggingFace + advanced)
    "StreamingDataset",
    "StreamingConfig",
    "MultiSourceDataset",
    "DocumentBuffer",
    "SequencePacker",
    "FineWebEduConfig",
    "OSCARConfig",
    "WikipediaDEConfig",
    "create_streaming_dataloader",
    "create_bilingual_dataloader",
    "TokenizerProtocol",
]
