"""Mini-batch helpers for causal language-model datasets."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset


def make_dataloader(
    dataset: Dataset[torch.Tensor],
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader[torch.Tensor]:
    """Create a DataLoader for token windows."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
