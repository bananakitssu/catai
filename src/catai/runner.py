"""High-level training runner for CatAI models."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import TrainingConfig
from .training import train_epochs


def run_training(
    model: nn.Module,
    loader: DataLoader[torch.Tensor],
    config: TrainingConfig,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    device: str | torch.device | None = None,
) -> list[float]:
    """Train a model using a shared configuration and return epoch losses."""
    if device is not None:
        model.to(torch.device(device))

    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    return train_epochs(
        model,
        optimizer,
        loader,
        epochs=config.epochs,
        grad_clip=config.grad_clip,
    )
