"""Training utilities for causal language modeling."""

from __future__ import annotations

import torch
from torch import nn


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


def train_step(model: nn.Module, optimizer: torch.optim.Optimizer, tokens: torch.Tensor) -> float:
    """Run one optimization step and return the loss before the update."""
    model.train()
    inputs, targets = make_next_token_batch(tokens)
    logits = model(inputs)
    loss = causal_language_model_loss(logits, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.detach())
