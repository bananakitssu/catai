"""High-level training runner for CatAI models."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from .config import TrainingConfig
from .training import train_epochs


def train_model(
    model: torch.nn.Module,
    loader: DataLoader[torch.Tensor],
    config: TrainingConfig,
    *,
    device: torch.device | str | None = None,
) -> list[float]:
    """Train a model with a TrainingConfig and return one loss per epoch."""
    target_device = torch.device(device) if device is not None else torch.device("cpu")
    model.to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    def batches_on_device() -> DataLoader[torch.Tensor]:
        class DeviceLoader:
            def __iter__(self):
                for batch in loader:
                    yield batch.to(target_device)

            def __len__(self):
                return len(loader)

        return DeviceLoader()  # type: ignore[return-value]

    return train_epochs(
        model,
        optimizer,
        batches_on_device(),
        epochs=config.epochs,
        grad_clip=config.grad_clip,
    )
