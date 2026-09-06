"""Training utilities for causal language modeling."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


def make_next_token_batch(tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split token sequences into inputs and one-token-shifted targets."""
    if tokens.ndim != 2 or tokens.size(1) < 2:
        raise ValueError("tokens must have shape (batch, sequence) with sequence >= 2")
    return tokens[:, :-1], tokens[:, 1:]


def causal_language_model_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute cross-entropy over next-token predictions."""
    if logits.ndim != 3 or targets.ndim != 2:
        raise ValueError("logits must be (batch, sequence, vocab) and targets must be (batch, sequence)")
    if logits.shape[:2] != targets.shape:
        raise ValueError("logits and targets sequence dimensions must match")
    return nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def clip_gradients(model: nn.Module, max_norm: float) -> float:
    """Clip model gradients to a maximum global norm and return the pre-clip norm."""
    if max_norm <= 0:
        raise ValueError("max_norm must be positive")
    return float(nn.utils.clip_grad_norm_(model.parameters(), max_norm))


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    tokens: torch.Tensor,
    grad_clip: float | None = None,
) -> float:
    """Run one optimization step and return the loss before the update."""
    if grad_clip is not None and grad_clip <= 0:
        raise ValueError("grad_clip must be positive when provided")
    model.train()
    inputs, targets = make_next_token_batch(tokens)
    logits = model(inputs)
    loss = causal_language_model_loss(logits, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if grad_clip is not None:
        clip_gradients(model, grad_clip)
    optimizer.step()
    return float(loss.detach())


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    tokens: torch.Tensor,
    steps: int,
    grad_clip: float | None = None,
) -> list[float]:
    """Train on a fixed token batch for a number of optimization steps."""
    if steps < 0:
        raise ValueError("steps must be non-negative")
    return [train_step(model, optimizer, tokens, grad_clip=grad_clip) for _ in range(steps)]


def train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader[torch.Tensor],
    grad_clip: float | None = None,
) -> float:
    """Train over every token window once and return the mean loss."""
    model.train()
    total_loss = 0.0
    batches = 0
    for tokens in loader:
        total_loss += train_step(model, optimizer, tokens, grad_clip=grad_clip)
        batches += 1
    if batches == 0:
        raise ValueError("loader must contain at least one batch")
    return total_loss / batches


def train_epochs(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader[torch.Tensor],
    epochs: int,
    grad_clip: float | None = None,
) -> list[float]:
    """Train over a DataLoader for multiple epochs and return epoch losses."""
    if epochs < 1:
        raise ValueError("epochs must be positive")
    return [train_epoch(model, optimizer, loader, grad_clip=grad_clip) for _ in range(epochs)]
