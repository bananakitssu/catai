"""Dataset helpers for causal language-model training."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class TokenWindowDataset(Dataset[torch.Tensor]):
    """Expose token windows with a configurable stride, including one target token."""

    def __init__(self, tokens: torch.Tensor, sequence_length: int, stride: int = 1) -> None:
        if tokens.ndim != 1:
            raise ValueError("tokens must have shape (sequence,)")
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        if stride < 1:
            raise ValueError("stride must be positive")
        if tokens.numel() <= sequence_length:
            raise ValueError("tokens must contain more than sequence_length tokens")
        self.tokens = tokens
        self.sequence_length = sequence_length
        self.stride = stride

    def __len__(self) -> int:
        available = self.tokens.numel() - self.sequence_length
        return max(0, (available + self.stride - 1) // self.stride)

    def __getitem__(self, index: int) -> torch.Tensor:
        if index < 0 or index >= len(self):
            raise IndexError("dataset index out of range")
        start = index * self.stride
        return self.tokens[start : start + self.sequence_length + 1]
