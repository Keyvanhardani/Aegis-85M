# EFC-Core Source Package
"""
EFC-Core: Efficient Factorized Core LLM
BitNet b1.58 + Mamba-2 Hybrid Architecture
"""

__version__ = "0.1.0"
__author__ = "EFC-Engineer"

from .model import EFCModel, EFCBlock, SlidingWindowAttention, create_efc_model
from .layers import BitLinear, RMSNorm, SelectiveSSM, MambaBlock
from .data import CharTokenizer, TextDataset, TinyShakespeare

__all__ = [
    # Model
    "EFCModel",
    "EFCBlock",
    "SlidingWindowAttention",
    "create_efc_model",
    # Layers
    "BitLinear",
    "RMSNorm",
    "SelectiveSSM",
    "MambaBlock",
    # Data
    "CharTokenizer",
    "TextDataset",
    "TinyShakespeare",
]
