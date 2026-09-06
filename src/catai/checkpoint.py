"""Checkpoint helpers for CatAI training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    *,
    epoch: int = 0,
    loss: float | None = None,
) -> None:
    """Save model, optimizer, and resumable training metadata."""
    if step < 0:
        raise ValueError("step must be non-negative")
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "epoch": epoch,
        "loss": loss,
    }
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> int:
    """Load a checkpoint and return its training step."""
    checkpoint: dict[str, Any] = torch.load(Path(path), map_location=map_location, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint["step"])


def load_training_metadata(path: str | Path) -> dict[str, int | float | None]:
    """Load resumable epoch and loss metadata without loading a model."""
    checkpoint: dict[str, Any] = torch.load(Path(path), map_location="cpu", weights_only=True)
    return {
        "step": int(checkpoint["step"]),
        "epoch": int(checkpoint.get("epoch", 0)),
        "loss": checkpoint.get("loss"),
    }
