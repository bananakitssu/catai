"""Autoregressive text generation utilities."""

from __future__ import annotations

import torch

from .model import CatAI


def generate(
    model: CatAI,
    tokens: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    repetition_penalty: float = 1.1,
    no_repeat_ngram_size: int = 3,
) -> torch.Tensor:
    """Generate tokens autoregressively from a prompt."""
    if tokens.ndim != 2:
        raise ValueError("tokens must have shape (batch, sequence)")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if repetition_penalty < 1.0:
        raise ValueError("repetition_penalty must be at least 1.0")
    if no_repeat_ngram_size < 0:
        raise ValueError("no_repeat_ngram_size must be non-negative")

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            result = tokens
            for _ in range(max_new_tokens):
                context = result[:, -model.max_seq_len :]
                logits = model(context)[:, -1, :] / temperature

                if repetition_penalty > 1.0:
                    for batch_index in range(result.size(0)):
                        seen = torch.unique(context[batch_index])
                        seen_logits = logits[batch_index, seen]
                        logits[batch_index, seen] = torch.where(
                            seen_logits < 0,
                            seen_logits * repetition_penalty,
                            seen_logits / repetition_penalty,
                        )

                if no_repeat_ngram_size > 1:
                    for batch_index in range(result.size(0)):
                        sequence = context[batch_index].tolist()
                        if len(sequence) >= no_repeat_ngram_size - 1:
                            prefix = tuple(sequence[-(no_repeat_ngram_size - 1) :])
                            banned = {
                                ngram[-1]
                                for ngram in (
                                    tuple(sequence[i : i + no_repeat_ngram_size])
                                    for i in range(len(sequence) - no_repeat_ngram_size + 1)
                                )
                                if ngram[:-1] == prefix
                            }
                            if banned:
                                logits[batch_index, list(banned)] = float("-inf")

                next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
                result = torch.cat((result, next_token), dim=1)
            return result
    finally:
        model.train(was_training)
