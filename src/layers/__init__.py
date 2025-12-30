# EFC-Core Layers Package
"""
Core neural network layers for EFC architecture.
- BitLinear: 1.58-bit ternary quantization
- RMSNorm: Root Mean Square normalization
- SSM: State Space Model (Mamba-style)
"""

from .bitlinear import BitLinear
from .rmsnorm import RMSNorm
from .ssm import SelectiveSSM, MambaBlock

__all__ = ["BitLinear", "RMSNorm", "SelectiveSSM", "MambaBlock"]
