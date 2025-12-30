"""
CPU-optimized EFC model configurations.

Three tiers designed for different CPU constraints:
- EFC-Tiny (8M): Mobile, embedded, Raspberry Pi
- EFC-Small (17M): Laptops, workstations
- EFC-Base (32M): High-performance servers

Based on research from:
- Microsoft bitnet.cpp (arxiv:2410.16144)
- Mamba SSM efficiency studies (arxiv:2312.00752)
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CPUModelConfig:
    """Configuration for CPU-optimized EFC models."""

    # Model architecture
    d_model: int
    n_layers: int
    n_heads: int
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    vocab_size: int = 23139  # BPE tokenizer vocab
    max_seq_len: int = 512
    attention_layers: List[int] = field(default_factory=list)
    window_size: int = 128

    # CPU optimization flags
    use_packed_weights: bool = True  # Pack ternary to INT2
    use_simd: bool = True  # Enable SIMD vectorization hints
    cache_ssm_state: bool = True  # Cache SSM state between calls

    @property
    def estimated_params(self) -> int:
        """Estimate parameter count."""
        # Embedding + LM head (tied)
        emb = self.vocab_size * self.d_model

        # Per layer params
        # SSM: input proj, conv, dt/B/C proj, output proj
        ssm_params = (
            self.d_model * self.d_model * self.expand * 2 +  # in/out proj
            self.d_model * self.expand * self.d_conv +  # conv
            self.d_model * self.expand * (self.d_state * 2 + 1)  # dt/B/C
        )

        # Attention (for attention layers)
        attn_params = 4 * self.d_model * self.d_model

        # BitLinear MLP: 2 linear layers with expansion
        mlp_params = 2 * self.d_model * (self.d_model * self.expand)

        # RMSNorm: 2 per layer
        norm_params = 2 * self.d_model

        # Total per layer (SSM-based)
        ssm_layer = ssm_params + mlp_params + norm_params
        attn_layer = attn_params + mlp_params + norm_params

        total = emb
        for i in range(self.n_layers):
            if i in self.attention_layers:
                total += attn_layer
            else:
                total += ssm_layer

        # Final norm
        total += self.d_model

        return total

    @property
    def packed_size_mb(self) -> float:
        """Estimate packed model size in MB (ternary = 2 bits per weight)."""
        # Ternary weights: 2 bits per weight
        # 8 weights per 2 bytes = 4 weights per byte
        return self.estimated_params / 4 / 1024 / 1024

    @property
    def fp16_size_mb(self) -> float:
        """Estimate FP16 model size in MB."""
        return self.estimated_params * 2 / 1024 / 1024


# Pre-defined CPU model configurations
EFC_TINY = CPUModelConfig(
    d_model=256,
    n_layers=4,
    n_heads=4,
    d_state=8,
    d_conv=4,
    expand=2,
    vocab_size=23139,
    max_seq_len=256,
    attention_layers=[3],  # Only last layer uses attention
    window_size=64,
)

EFC_SMALL = CPUModelConfig(
    d_model=384,
    n_layers=6,
    n_heads=6,
    d_state=12,
    d_conv=4,
    expand=2,
    vocab_size=23139,
    max_seq_len=512,
    attention_layers=[5],  # Only last layer uses attention
    window_size=128,
)

EFC_BASE = CPUModelConfig(
    d_model=512,
    n_layers=8,
    n_heads=8,
    d_state=16,
    d_conv=4,
    expand=2,
    vocab_size=23139,
    max_seq_len=1024,
    attention_layers=[3, 7],  # Two attention layers
    window_size=256,
)

EFC_FULL = CPUModelConfig(
    d_model=768,
    n_layers=12,
    n_heads=12,
    d_state=16,
    d_conv=4,
    expand=2,
    vocab_size=23139,
    max_seq_len=2048,
    attention_layers=[3, 7, 11],  # Every 4th layer
    window_size=256,
    use_packed_weights=False,  # Full GPU model
    use_simd=False,
)


def get_config(tier: str) -> CPUModelConfig:
    """Get configuration by tier name."""
    configs = {
        'tiny': EFC_TINY,
        'small': EFC_SMALL,
        'base': EFC_BASE,
        'full': EFC_FULL,
    }
    if tier.lower() not in configs:
        raise ValueError(f"Unknown tier: {tier}. Available: {list(configs.keys())}")
    return configs[tier.lower()]


def print_config_summary():
    """Print summary of all CPU model configurations."""
    print("=" * 70)
    print("EFC-Core CPU Model Configurations")
    print("=" * 70)

    for name, config in [
        ("EFC-Tiny", EFC_TINY),
        ("EFC-Small", EFC_SMALL),
        ("EFC-Base", EFC_BASE),
        ("EFC-Full (GPU)", EFC_FULL),
    ]:
        params = config.estimated_params
        print(f"\n{name}:")
        print(f"  Architecture: d={config.d_model}, L={config.n_layers}, H={config.n_heads}")
        print(f"  Parameters: {params:,} ({params/1e6:.2f}M)")
        print(f"  Packed size: {config.packed_size_mb:.2f} MB (ternary)")
        print(f"  FP16 size: {config.fp16_size_mb:.2f} MB")
        print(f"  Max seq len: {config.max_seq_len}")
        print(f"  Attention layers: {config.attention_layers}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    print_config_summary()
