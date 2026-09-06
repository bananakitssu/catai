"""Dataset helpers for causal language-model training."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class TokenWindowDataset(Dataset[torch.Tensor]):
    """Expose overlapping token windows, including one target token."""

    def __init__(self, tokens: torch.Tensor, sequence_length: int) -> None:
        if tokens.ndim != 1:
            raise ValueError("tokens must have shape (sequence,)")
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        if tokens.numel() <= sequence_length:
            raise ValueError("tokens must contain more than sequence_length tokens")
        self.tokens = tokens
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return self.tokens.numel() - self.sequence_length

    def __getitem__(self, index: int) -> torch.Tensor:
        if index < 0 or index >= len(self):
            raise IndexError("dataset index out of range")
        return self.tokens[index : index + self.sequence_length + 1]
