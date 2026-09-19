"""A modern decoder-only Transformer language model for CatAI.

Uses:
- Rotary Position Embeddings (RoPE)
- RMSNorm
- SwiGLU feed-forward
- Causal multi-head self-attention
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors."""
    # q, k: (batch, heads, seq, head_dim)
    # cos, sin: (1, 1, seq, head_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    """Precomputes cos/sin caches for RoPE."""

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE dimension must be even")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # (seq, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (seq, dim)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.cos_cached.size(2):
            self._build_cache(seq_len)
        return self.cos_cached[:, :, :seq_len, :], self.sin_cached[:, :, :seq_len, :]


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE."""

    def __init__(self, d_model: int, n_heads: int, context_length: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=context_length)

        # Causal mask buffer (True = masked)
        mask = torch.triu(torch.ones(context_length, context_length, dtype=torch.bool), diagonal=1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        if seq_len > self.mask.size(0):
            raise ValueError("sequence exceeds context_length")

        qkv = self.qkv(x)
        qkv = qkv.view(batch, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        cos, sin = self.rope(seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.masked_fill(self.mask[:seq_len, :seq_len], float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (batch, heads, seq, head_dim)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.out_proj(out)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network (used by LLaMA-style models)."""

    def __init__(self, d_model: int, hidden_dim: int | None = None, dropout: float = 0.0) -> None:
        super().__init__()
        # Standard SwiGLU expansion is roughly 8/3 * d_model, rounded to multiple of 256 for larger models.
        if hidden_dim is None:
            hidden_dim = int(8 * d_model / 3)
            # Keep it a multiple of 64 for better kernel utilization on small models too.
            hidden_dim = ((hidden_dim + 63) // 64) * 64
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, context_length: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attention = CausalSelfAttention(d_model, n_heads, context_length, dropout)
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class CatAI(nn.Module):
    """Decoder-only Transformer for causal language modeling."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 256,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.0,
        *,
        context_length: int | None = None,
    ) -> None:
        super().__init__()
        if max_seq_len < 1:
            raise ValueError("max_seq_len must be positive")
        if context_length is not None:
            if context_length < 1:
                raise ValueError("context_length must be positive")
            max_seq_len = context_length
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.context_length = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, self.context_length, dropout=dropout) for _ in range(n_layers)]
        )
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_embedding.weight

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape (batch, sequence)")
        batch, length = tokens.shape
        if length > self.context_length:
            raise ValueError("sequence exceeds context_length")

        x = self.token_embedding(tokens)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.norm(x))

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
