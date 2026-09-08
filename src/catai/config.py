"""Configuration for CatAI training."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    """Basic settings shared by CatAI training entry points."""

    batch_size: int = 32
    learning_rate: float = 3e-4
    epochs: int = 1
    grad_clip: float | None = 1.0
    validation_split: float = 0.0
    patience: int | None = None

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.grad_clip is not None and self.grad_clip <= 0:
            raise ValueError("grad_clip must be positive when provided")
        if not 0.0 <= self.validation_split < 1.0:
            raise ValueError("validation_split must be in [0, 1)")
        if self.patience is not None and self.patience < 1:
            raise ValueError("patience must be positive when provided")
