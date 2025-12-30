"""
CPU inference utilities for EFC models.

Implements:
- Ternary weight packing (32x compression from FP32)
- Weight unpacking for inference
- SSM state caching for streaming generation
- Optimized inference functions

Based on Microsoft bitnet.cpp techniques.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class PackedWeights:
    """Container for packed ternary weights."""
    packed_data: torch.Tensor  # Packed INT8 tensor (4 weights per byte)
    scale: float  # Original weight scale for reconstruction
    shape: Tuple[int, ...]  # Original shape


def pack_ternary_weights(weights: torch.Tensor) -> PackedWeights:
    """
    Pack ternary weights {-1, 0, +1} to 2-bit representation.

    Packing scheme:
    - 00 = -1
    - 01 = 0
    - 10 = +1
    - 4 weights packed per byte

    Args:
        weights: Tensor with values in {-1, 0, +1}

    Returns:
        PackedWeights with compressed data
    """
    # Store original shape
    original_shape = weights.shape
    flat = weights.flatten()

    # Ensure length is multiple of 4 (pad if needed)
    pad_len = (4 - len(flat) % 4) % 4
    if pad_len > 0:
        flat = torch.cat([flat, torch.zeros(pad_len, dtype=flat.dtype)])

    # Convert {-1, 0, +1} to {0, 1, 2}
    mapped = (flat + 1).to(torch.uint8)

    # Reshape for packing (groups of 4)
    grouped = mapped.view(-1, 4)

    # Pack 4 values into 1 byte
    # Each value uses 2 bits: v0 | (v1 << 2) | (v2 << 4) | (v3 << 6)
    packed = (grouped[:, 0] |
              (grouped[:, 1] << 2) |
              (grouped[:, 2] << 4) |
              (grouped[:, 3] << 6))

    # Calculate scale (for activation quantization compatibility)
    scale = weights.abs().mean().item() if weights.numel() > 0 else 1.0

    return PackedWeights(
        packed_data=packed,
        scale=scale,
        shape=original_shape
    )


def unpack_ternary_weights(packed: PackedWeights) -> torch.Tensor:
    """
    Unpack ternary weights from 2-bit representation.

    Args:
        packed: PackedWeights container

    Returns:
        Unpacked tensor with values in {-1, 0, +1}
    """
    data = packed.packed_data

    # Extract 4 values from each byte
    v0 = data & 0x03  # bits 0-1
    v1 = (data >> 2) & 0x03  # bits 2-3
    v2 = (data >> 4) & 0x03  # bits 4-5
    v3 = (data >> 6) & 0x03  # bits 6-7

    # Stack and flatten
    unpacked = torch.stack([v0, v1, v2, v3], dim=-1).flatten()

    # Convert {0, 1, 2} back to {-1, 0, +1}
    result = unpacked.to(torch.float32) - 1.0

    # Trim to original size and reshape
    total_elements = 1
    for dim in packed.shape:
        total_elements *= dim

    result = result[:total_elements].view(packed.shape)

    return result


def pack_model_weights(model: nn.Module) -> Dict[str, PackedWeights]:
    """
    Pack all ternary weights in a model.

    Args:
        model: PyTorch model with BitLinear layers

    Returns:
        Dictionary mapping parameter names to packed weights
    """
    packed = {}

    for name, param in model.named_parameters():
        # Check if this is a BitLinear weight (not bias, not embedding)
        if 'weight' in name and 'bit_linear' in name.lower():
            # Quantize to ternary and pack
            with torch.no_grad():
                # Apply ternary quantization
                alpha = param.data.abs().mean()
                ternary = torch.sign(param.data) * (param.data.abs() > alpha * 0.3).float()

                packed[name] = pack_ternary_weights(ternary)
                print(f"  Packed {name}: {param.shape} -> {packed[name].packed_data.shape}")

    return packed


class SSMStateCache:
    """
    Cache for SSM hidden states during streaming inference.

    Maintains constant memory regardless of sequence length.
    """

    def __init__(self, batch_size: int, n_layers: int, d_model: int, d_state: int):
        """
        Initialize SSM state cache.

        Args:
            batch_size: Batch size for inference
            n_layers: Number of model layers
            d_model: Model dimension
            d_state: SSM state dimension
        """
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_state = d_state

        # State shape: [batch, n_layers, d_model, d_state]
        # For EFC-Small: 1 * 6 * 384 * 12 * 4 bytes = 110 KB
        self.state = torch.zeros(batch_size, n_layers, d_model, d_state)

        # Position counter for each batch item
        self.position = torch.zeros(batch_size, dtype=torch.long)

    def get_state(self, layer_idx: int) -> torch.Tensor:
        """Get SSM state for a specific layer."""
        return self.state[:, layer_idx]

    def update_state(self, layer_idx: int, new_state: torch.Tensor):
        """Update SSM state for a specific layer."""
        self.state[:, layer_idx] = new_state

    def increment_position(self, n_tokens: int = 1):
        """Increment position counter."""
        self.position += n_tokens

    def reset(self):
        """Reset all states to zero."""
        self.state.zero_()
        self.position.zero_()

    @property
    def memory_bytes(self) -> int:
        """Calculate memory usage in bytes."""
        state_bytes = self.state.numel() * self.state.element_size()
        pos_bytes = self.position.numel() * self.position.element_size()
        return state_bytes + pos_bytes


def estimate_cpu_throughput(config, cpu_cores: int = 4) -> dict:
    """
    Estimate CPU inference throughput.

    Based on bitnet.cpp benchmarks:
    - Modern x86 (AVX2): 2.37x-6.17x speedup over baseline
    - ARM (NEON): 1.37x-5.07x speedup

    Args:
        config: CPUModelConfig
        cpu_cores: Number of CPU cores available

    Returns:
        Dictionary with throughput estimates
    """
    params = config.estimated_params

    # Baseline: FP16 inference on single core
    # Roughly 1 GFLOP/s per modern core
    gflops_per_core = 1.0

    # Operations per token (rough estimate)
    # 2 * params FLOPs for forward pass
    ops_per_token = 2 * params

    # Baseline single-core throughput
    baseline_tps = gflops_per_core * 1e9 / ops_per_token

    # BitNet speedup factors
    x86_speedup = 4.0  # Middle of 2.37-6.17 range
    arm_speedup = 3.0  # Middle of 1.37-5.07 range

    # Parallel scaling (not perfect, ~70% efficiency)
    parallel_factor = 1 + (cpu_cores - 1) * 0.7

    return {
        'baseline_single_core': baseline_tps,
        'x86_estimate': baseline_tps * x86_speedup * parallel_factor,
        'arm_estimate': baseline_tps * arm_speedup * parallel_factor,
        'cores': cpu_cores,
        'params_millions': params / 1e6,
    }


def verify_packing_roundtrip():
    """Verify that packing/unpacking preserves ternary values."""
    print("Verifying ternary packing roundtrip...")

    # Test various shapes
    shapes = [(64,), (128, 256), (512, 768), (12, 768, 768)]

    for shape in shapes:
        # Create random ternary tensor
        original = torch.randint(-1, 2, shape).float()

        # Pack and unpack
        packed = pack_ternary_weights(original)
        unpacked = unpack_ternary_weights(packed)

        # Verify
        matches = torch.allclose(original, unpacked)
        compression = original.numel() * 4 / packed.packed_data.numel()

        print(f"  Shape {shape}: {'PASS' if matches else 'FAIL'}, "
              f"compression {compression:.1f}x")

    print("Roundtrip verification complete.\n")


if __name__ == "__main__":
    verify_packing_roundtrip()

    # Test SSM state cache
    print("Testing SSM state cache...")
    cache = SSMStateCache(batch_size=1, n_layers=6, d_model=384, d_state=12)
    print(f"  Memory usage: {cache.memory_bytes / 1024:.2f} KB")

    # Test throughput estimation
    from src.configs.cpu_models import EFC_SMALL, print_config_summary
    print("\nEstimating CPU throughput for EFC-Small...")
    estimate = estimate_cpu_throughput(EFC_SMALL, cpu_cores=4)
    print(f"  Parameters: {estimate['params_millions']:.2f}M")
    print(f"  x86 estimate: {estimate['x86_estimate']:.1f} tok/s (4 cores)")
    print(f"  ARM estimate: {estimate['arm_estimate']:.1f} tok/s (4 cores)")
