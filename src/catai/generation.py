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
    eos_token_id: int | None = None,
    top_k: int | None = None,
) -> torch.Tensor:
    """Generate tokens autoregressively with optional top-k sampling and EOS stopping."""
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
    if top_k is not None and (top_k < 1 or top_k > model.vocab_size):
        raise ValueError("top_k must be between 1 and the model vocabulary size")
    if eos_token_id is not None and (eos_token_id < 0 or eos_token_id >= model.vocab_size):
        raise ValueError("eos_token_id outside vocabulary")

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            result = tokens
            finished = torch.zeros(result.size(0), dtype=torch.bool, device=result.device)
            for _ in range(max_new_tokens):
                context = result[:, -model.context_length :]
                logits = model(context)[:, -1, :] / temperature

                if repetition_penalty > 1.0:
                    for batch_index in range(result.size(0)):
                        seen = torch.unique(context[batch_index])
                        if eos_token_id is not None:
                            seen = seen[seen != eos_token_id]
                        seen_logits = logits[batch_index, seen]
                        logits[batch_index, seen] = torch.where(
                            seen_logits < 0,
                            seen_logits * repetition_penalty,
                            seen_logits / repetition_penalty,
                        )

                if eos_token_id is not None:
                    logits[:, eos_token_id] = torch.where(
                        finished,
                        torch.tensor(float("-inf"), device=logits.device),
                        logits[:, eos_token_id],
                    )

                if no_repeat_ngram_size > 1:
                    for batch_index in range(result.size(0)):
                        if finished[batch_index]:
                            continue
                        sequence = context[batch_index].tolist()
                        if len(sequence) >= no_repeat_ngram_size - 1:
                            prefix = tuple(sequence[-(no_repeat_ngram_size - 1) :])
                            banned = {
                                ngram[-1]
                                for ngram in (
                                    tuple(sequence[i : i + no_repeat_ngram_size])
                                    for i in range(len(sequence) - no_repeat_ngram_size + 1)
                                )
                                if ngram[:-1] == prefix and ngram[-1] != eos_token_id
                            }
                            if banned:
                                logits[batch_index, list(banned)] = float("-inf")

                if top_k is not None:
                    values, indices = torch.topk(logits, top_k, dim=-1)
                    filtered = torch.full_like(logits, float("-inf"))
                    filtered.scatter_(1, indices, values)
                    logits = filtered

                next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
                if eos_token_id is not None:
                    next_token = torch.where(
                        finished.unsqueeze(1),
                        torch.full_like(next_token, eos_token_id),
                        next_token,
                    )
                    finished |= next_token.squeeze(1).eq(eos_token_id)
                result = torch.cat((result, next_token), dim=1)
                if eos_token_id is not None and bool(finished.all()):
                    break
            return result
    finally:
        model.train(was_training)
