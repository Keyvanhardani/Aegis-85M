# BitLinear Layer Implementation
# BitNet b1.58 - Ternary Weight Quantization with STE
# Reference: "The Era of 1-bit LLMs" (Ma et al., 2024)

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class StraightThroughEstimator(torch.autograd.Function):
    """
    Straight-Through Estimator for quantization.
    Forward: Returns quantized tensor.
    Backward: Passes gradients unchanged (as if no quantization).
    """

    @staticmethod
    def forward(ctx, x: Tensor, quantized: Tensor) -> Tensor:
        return quantized

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        # Gradient passes through unchanged
        return grad_output, None


def ste_quantize(x: Tensor, quantized: Tensor) -> Tensor:
    """Apply STE: forward uses quantized, backward uses original."""
    return StraightThroughEstimator.apply(x, quantized)


def ternarize_weights(weights: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Ternarize weights to {-1, 0, +1} using BitNet b1.58 method.

    Method:
    1. Compute scale factor gamma = mean(|W|)
    2. Normalize: W_norm = W / gamma
    3. Round and clip to {-1, 0, +1}

    Args:
        weights: Full-precision weight tensor
        eps: Small constant for numerical stability

    Returns:
        Ternary weights in {-1, 0, +1}
    """
    # Compute per-tensor scale factor (mean absolute value)
    gamma = weights.abs().mean() + eps

    # Normalize and round to nearest integer, clip to [-1, 1]
    w_normalized = weights / gamma
    w_ternary = torch.clamp(torch.round(w_normalized), min=-1.0, max=1.0)

    return w_ternary


def quantize_activations(
    x: Tensor,
    num_bits: int = 8,
    eps: float = 1e-8
) -> tuple[Tensor, Tensor]:
    """
    Quantize activations using absmax scaling (BitNet style).

    Method:
    1. Compute Qb = 2^(b-1) for b-bit quantization
    2. Scale = Qb / max(|x|)
    3. x_quant = round(clip(x * scale, -Qb, Qb))

    Args:
        x: Input activation tensor
        num_bits: Number of bits for quantization (default 8)
        eps: Numerical stability constant

    Returns:
        Tuple of (quantized_activations, scale_factor)
    """
    qb = 2 ** (num_bits - 1)  # 128 for 8-bit

    # Compute absmax scale per batch element
    # Keep dimensions for broadcasting
    absmax = x.abs().amax(dim=-1, keepdim=True) + eps
    scale = qb / absmax

    # Quantize
    x_scaled = x * scale
    x_quant = torch.clamp(torch.round(x_scaled), min=-qb, max=qb)

    return x_quant, scale


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    More efficient than LayerNorm (no mean subtraction).

    Formula: x * rsqrt(mean(x^2) + eps) * weight
    """

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        # Compute RMS
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        x_normed = x / rms
        return x_normed * self.weight


class BitLinear(nn.Module):
    """
    BitLinear layer with 1.58-bit ternary weight quantization.

    Key features:
    - Weights stored in full precision, quantized on-the-fly during forward
    - Ternary weights {-1, 0, +1} for minimal memory and compute
    - Activation quantization (8-bit) before matmul
    - Pre-RMSNorm for distribution stability
    - Straight-Through Estimator for gradient flow

    Args:
        in_features: Input dimension
        out_features: Output dimension
        bias: Whether to include bias (default False per BitNet)
        num_bits: Activation quantization bits (default 8)
        eps: Numerical stability constant
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        num_bits: int = 8,
        eps: float = 1e-8
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_bits = num_bits
        self.eps = eps

        # Full precision weights (quantized during forward)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

        # Optional bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        # Pre-RMSNorm for activation stability
        self.norm = RMSNorm(in_features, eps=eps)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with scaled normal distribution."""
        # Use scaled initialization suitable for ternary quantization
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def get_ternary_weights(self) -> Tensor:
        """Get the current ternary weights (for inference/debugging)."""
        return ternarize_weights(self.weight, self.eps)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with quantization.

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            Output tensor of shape (..., out_features)
        """
        # Step 1: Pre-RMSNorm for activation stability
        x_normed = self.norm(x)

        # Step 2: Quantize activations (8-bit absmax)
        x_quant, scale = quantize_activations(x_normed, self.num_bits, self.eps)

        # Apply STE for activation quantization
        x_q = ste_quantize(x_normed, x_quant / scale)  # Dequantize for computation

        # Step 3: Ternarize weights
        w_ternary = ternarize_weights(self.weight, self.eps)

        # Apply STE for weight quantization
        w_q = ste_quantize(self.weight, w_ternary)

        # Step 4: Compute output (F.linear for efficiency)
        # Scale factor for ternary weights: multiply by gamma
        gamma = self.weight.abs().mean() + self.eps
        output = F.linear(x_q, w_q) * gamma

        # Step 5: Add bias if present
        if self.bias is not None:
            output = output + self.bias

        return output

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"num_bits={self.num_bits}"
        )
