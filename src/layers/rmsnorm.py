# RMSNorm Layer Implementation
# Root Mean Square Layer Normalization for EFC Architecture
# More efficient than LayerNorm (no mean computation required)

import torch
import torch.nn as nn
from torch import Tensor


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Advantages over LayerNorm:
    - No mean subtraction required
    - Fewer operations (faster on CPU/GPU)
    - Empirically equivalent quality in LLMs

    Formula: x_norm = x * rsqrt(mean(x^2) + eps) * weight

    Args:
        dim: Feature dimension to normalize over (last axis)
        eps: Small constant for numerical stability
    """

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: Tensor) -> Tensor:
        """Compute RMS normalization without learned scale."""
        # Compute mean of squares along last dimension
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply RMS normalization.

        Args:
            x: Input tensor of shape (..., dim)

        Returns:
            Normalized tensor of same shape
        """
        # Normalize
        x_normed = self._norm(x.float())

        # Apply learned scale and cast back to input dtype
        return (x_normed * self.weight).type_as(x)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}"
