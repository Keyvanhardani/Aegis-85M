# EFC-Core Model Implementation
# Hybrid BitNet b1.58 + Mamba-2 Architecture
# Reference: Architecture spec (docs/architecture.md)

from typing import Optional, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .layers.bitlinear import BitLinear
from .layers.rmsnorm import RMSNorm
from .layers.ssm import SelectiveSSM


class SlidingWindowAttention(nn.Module):
    """
    Sliding Window Attention for local context recall.

    Used every 4th layer to complement SSM's implicit memory with
    explicit local attention. More efficient than full attention
    for long sequences: O(L * W) vs O(L^2).

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        window_size: Size of sliding window (tokens on each side)
        dropout: Attention dropout probability
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window_size: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.window_size = window_size
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        # QKV projection (combined for efficiency)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def _create_sliding_window_mask(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> Tensor:
        """
        Create causal sliding window attention mask.

        For query position i attending to key position j:
        - Block if j > i (future tokens - causal constraint)
        - Block if i - j > window_size (outside window)

        Returns:
            Mask tensor [seq_len, seq_len] where [i,j] masks query i attending to key j
        """
        # Create position indices
        # query_pos[i] = i, key_pos[j] = j
        query_pos = torch.arange(seq_len, device=device).unsqueeze(1)  # [L, 1]
        key_pos = torch.arange(seq_len, device=device).unsqueeze(0)    # [1, L]

        # Distance from query to key: positive means key is in the past
        # diff[i,j] = query_pos[i] - key_pos[j] = i - j
        diff = query_pos - key_pos

        # Valid attention: key is in past (diff >= 0) AND within window (diff <= window_size)
        # Block where: diff < 0 (future) OR diff > window_size (too far past)
        block_mask = (diff < 0) | (diff > self.window_size)

        # Convert to attention mask format (-inf for blocked positions)
        attn_mask = torch.where(
            block_mask,
            torch.tensor(float('-inf'), device=device, dtype=dtype),
            torch.tensor(0.0, device=device, dtype=dtype)
        )

        return attn_mask

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with sliding window attention.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape

        # QKV projection
        qkv = self.qkv_proj(x)  # [B, L, 3*D]
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, L, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each [B, H, L, head_dim]

        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply sliding window mask
        mask = self._create_sliding_window_mask(seq_len, x.device, x.dtype)
        attn_weights = attn_weights + mask.unsqueeze(0).unsqueeze(0)

        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # [B, H, L, head_dim]

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        output = self.out_proj(attn_output)

        return output

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, n_heads={self.n_heads}, "
            f"window_size={self.window_size}"
        )


class BitLinearMLP(nn.Module):
    """
    BitLinear MLP block with GELU activation.

    Architecture: Linear(d_model -> expand*d_model) -> GELU -> Linear(expand*d_model -> d_model)
    Uses BitLinear for efficient ternary quantization.

    Args:
        d_model: Model dimension
        expand: Expansion factor for hidden dimension
    """

    def __init__(self, d_model: int, expand: int = 4):
        super().__init__()

        self.d_model = d_model
        self.hidden_dim = d_model * expand

        self.up_proj = BitLinear(d_model, self.hidden_dim)
        self.down_proj = BitLinear(self.hidden_dim, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: up -> GELU -> down."""
        h = self.up_proj(x)
        h = F.gelu(h)
        return self.down_proj(h)


class EFCBlock(nn.Module):
    """
    Single EFC Transformer block.

    Architecture (per docs/architecture.md):
        x -> RMSNorm -> Mixer (SSM or Attention) -> Residual
          -> RMSNorm -> BitLinear MLP -> Residual -> output

    Args:
        d_model: Model dimension
        d_state: SSM state dimension (if using SSM mixer)
        d_conv: SSM convolution kernel size
        n_heads: Number of attention heads (if using attention mixer)
        expand: MLP expansion factor
        window_size: Attention window size
        use_attention: If True, use SlidingWindowAttention; else use SSM
        norm_eps: RMSNorm epsilon
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        n_heads: int = 12,
        expand: int = 4,
        window_size: int = 256,
        use_attention: bool = False,
        norm_eps: float = 1e-8,
    ):
        super().__init__()

        self.d_model = d_model
        self.use_attention = use_attention

        # Mixer pre-norm
        self.norm1 = RMSNorm(d_model, eps=norm_eps)

        # Mixer: SSM or Attention
        if use_attention:
            self.mixer = SlidingWindowAttention(
                d_model=d_model,
                n_heads=n_heads,
                window_size=window_size,
            )
        else:
            self.mixer = SelectiveSSM(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=2,  # SSM internal expansion
            )

        # MLP pre-norm
        self.norm2 = RMSNorm(d_model, eps=norm_eps)

        # MLP
        self.mlp = BitLinearMLP(d_model, expand=expand)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with residual connections.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        # Mixer path with residual
        h = x + self.mixer(self.norm1(x))

        # MLP path with residual
        out = h + self.mlp(self.norm2(h))

        return out

    def extra_repr(self) -> str:
        mixer_type = "Attention" if self.use_attention else "SSM"
        return f"d_model={self.d_model}, mixer={mixer_type}"


class EFCModel(nn.Module):
    """
    Complete EFC Language Model.

    Hybrid architecture combining:
    - BitNet b1.58 ternary quantization
    - Mamba-style SSM for efficient sequence mixing
    - Sliding Window Attention (every 4th layer) for explicit recall

    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension
        n_layers: Number of transformer blocks
        n_heads: Number of attention heads
        d_state: SSM state dimension
        d_conv: SSM convolution kernel size
        expand: MLP expansion factor
        max_seq_len: Maximum sequence length (for position embeddings)
        attention_layers: Layer indices that use attention instead of SSM
        window_size: Attention window size
        norm_eps: RMSNorm epsilon
        tie_weights: Whether to tie input/output embeddings
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 4,
        max_seq_len: int = 2048,
        attention_layers: Optional[List[int]] = None,
        window_size: int = 256,
        norm_eps: float = 1e-8,
        tie_weights: bool = True,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len

        # Default attention layers: every 4th layer
        if attention_layers is None:
            attention_layers = [i for i in range(n_layers) if (i + 1) % 4 == 0]
        self.attention_layers = set(attention_layers)

        # Token embedding
        self.embed_tokens = nn.Embedding(vocab_size, d_model)

        # Transformer blocks
        self.layers = nn.ModuleList([
            EFCBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                n_heads=n_heads,
                expand=expand,
                window_size=window_size,
                use_attention=(i in self.attention_layers),
                norm_eps=norm_eps,
            )
            for i in range(n_layers)
        ])

        # Final layer norm
        self.norm = RMSNorm(d_model, eps=norm_eps)

        # Output head (language modeling)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.tie_weights = tie_weights
        if tie_weights:
            self.lm_head.weight = self.embed_tokens.weight

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        # Embedding initialization
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=0.02)

        # LM head (if not tied)
        if not self.tie_weights:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: Tensor,
        labels: Optional[Tensor] = None,
    ) -> dict:
        """
        Forward pass for language modeling.

        Args:
            input_ids: Token IDs [batch, seq_len]
            labels: Target token IDs for loss computation [batch, seq_len]

        Returns:
            Dictionary with:
                - logits: Output logits [batch, seq_len, vocab_size]
                - loss: Cross-entropy loss (if labels provided)
        """
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        h = self.embed_tokens(input_ids)  # [B, L, D]

        # Apply transformer blocks
        for layer in self.layers:
            h = layer(h)

        # Final normalization
        h = self.norm(h)

        # LM head
        logits = self.lm_head(h)  # [B, L, vocab_size]

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            # Cross-entropy loss
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {"logits": logits, "loss": loss}

    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> Tensor:
        """
        Autoregressive text generation.

        Args:
            input_ids: Initial token IDs [batch, seq_len]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling (None for greedy)

        Returns:
            Generated token IDs [batch, seq_len + max_new_tokens]
        """
        for _ in range(max_new_tokens):
            # Crop to max_seq_len if needed
            input_cropped = input_ids[:, -self.max_seq_len:]

            # Forward pass
            outputs = self(input_cropped)
            logits = outputs["logits"]

            # Get logits for last position
            next_logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = float('-inf')

            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def count_parameters(self) -> dict:
        """Count trainable and total parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "total": total,
            "trainable": trainable,
            "total_millions": total / 1e6,
            "trainable_millions": trainable / 1e6,
        }

    def extra_repr(self) -> str:
        params = self.count_parameters()
        return (
            f"vocab_size={self.vocab_size}, d_model={self.d_model}, "
            f"n_layers={self.n_layers}, params={params['total_millions']:.1f}M"
        )


def create_efc_model(config: Optional[dict] = None) -> EFCModel:
    """
    Factory function to create EFC model from config.

    Args:
        config: Model configuration dict (uses defaults if None)

    Returns:
        Configured EFCModel instance
    """
    default_config = {
        "vocab_size": 32000,
        "d_model": 768,
        "n_layers": 12,
        "n_heads": 12,
        "d_state": 16,
        "d_conv": 4,
        "expand": 4,
        "max_seq_len": 2048,
        "attention_layers": [3, 7, 11],
        "window_size": 256,
    }

    if config is not None:
        default_config.update(config)

    return EFCModel(**default_config)
