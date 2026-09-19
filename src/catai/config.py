"""Configuration for CatAI training and model scaling."""

from __future__ import annotations

from dataclasses import dataclass


# (d_model, n_heads, n_layers)
MODEL_PRESETS: dict[str, tuple[int, int, int]] = {
    "tiny": (128, 4, 4),       # ~0.8–1M params (depends on vocab)
    "small": (256, 8, 8),      # a few million
    "base": (512, 8, 12),      # tens of millions
    "medium": (768, 12, 16),   # larger experimental size
}


@dataclass(frozen=True)
class ModelConfig:
    """Transformer dimensions for a named CatAI model size."""

    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4

    def __post_init__(self) -> None:
        if self.d_model < 1 or self.n_heads < 1 or self.n_layers < 1:
            raise ValueError("model dimensions must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

    @classmethod
    def from_preset(cls, name: str) -> "ModelConfig":
        try:
            d_model, n_heads, n_layers = MODEL_PRESETS[name]
        except KeyError as exc:
            raise ValueError(f"unknown model preset: {name!r}") from exc
        return cls(d_model=d_model, n_heads=n_heads, n_layers=n_layers)


@dataclass(frozen=True)
class TrainingConfig:
    """Basic settings shared by CatAI training entry points."""

    batch_size: int = 32
    learning_rate: float = 3e-4
    epochs: int = 1
    grad_clip: float | None = 1.0
    validation_split: float = 0.0
    patience: int | None = None
    max_tokens: int | None = None

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
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError("max_tokens must be positive when provided")
