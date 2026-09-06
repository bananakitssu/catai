"""Device selection helpers for CatAI."""

from __future__ import annotations

import torch


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Resolve an explicit device or choose the best available accelerator."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
