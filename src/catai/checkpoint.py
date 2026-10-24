"""Checkpoint helpers for CatAI training."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import os
import tempfile

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
    metadata: dict[str, Any] | None = None,
    extra_state: dict[str, Any] | None = None,
) -> None:
    """Save model, optimizer, resumable progress, and optional metadata."""
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
        "metadata": metadata or {},
        "extra_state": extra_state or {},
    }
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{checkpoint_path.name}.", suffix=".tmp", dir=checkpoint_path.parent
    )
    os.close(fd)
    try:
        torch.save(checkpoint, temporary_name)
        os.replace(temporary_name, checkpoint_path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


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


def load_checkpoint_metadata(path: str | Path) -> dict[str, Any]:
    """Load the optional model/tokenizer metadata from a checkpoint."""
    checkpoint: dict[str, Any] = torch.load(Path(path), map_location="cpu", weights_only=True)
    return dict(checkpoint.get("metadata", {}))


def load_checkpoint_state(path: str | Path) -> dict[str, Any]:
    """Load additional training state stored alongside model and optimizer weights."""
    checkpoint: dict[str, Any] = torch.load(Path(path), map_location="cpu", weights_only=True)
    return dict(checkpoint.get("extra_state", {}))
