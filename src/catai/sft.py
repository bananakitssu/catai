"""Supervised fine-tuning utilities for CatAI chat models."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.utils.data import Dataset


IGNORE_INDEX = -100
ROLE_HEADER = "<|{role}|>\n"


def chat_token_stream(
    messages: Sequence[dict[str, str]],
    tokenizer: object,
) -> tuple[list[int], list[bool]]:
    """Encode a chat and mark tokens that belong to assistant responses."""
    if not messages:
        raise ValueError("messages must not be empty")

    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise TypeError("tokenizer must provide an encode method")
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    token_ids: list[int] = []
    train_mask: list[bool] = []
    last_role = None

    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported message role: {role!r}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("message content must be a non-empty string")

        header_tokens = list(encode(ROLE_HEADER.format(role=role)))
        content_tokens = list(encode(content))
        newline_tokens = list(encode("\n"))

        token_ids.extend(header_tokens)
        train_mask.extend([False] * len(header_tokens))

        token_ids.extend(content_tokens)
        train_mask.extend([role == "assistant"] * len(content_tokens))

        token_ids.extend(newline_tokens)
        train_mask.extend([role == "assistant"] * len(newline_tokens))
        last_role = role

    if eos_token_id is not None:
        token_ids.append(int(eos_token_id))
        train_mask.append(last_role == "assistant")

    if not token_ids:
        raise ValueError("chat messages must encode to at least one token")
    if not any(train_mask):
        raise ValueError("chat must contain assistant tokens")

    return token_ids, train_mask


class ChatSupervisedDataset(Dataset[dict[str, torch.Tensor]]):
    """Fixed-length chat examples with loss masked to assistant responses."""

    def __init__(
        self,
        examples: Sequence[Sequence[dict[str, str]]],
        tokenizer: object,
        sequence_length: int,
    ) -> None:
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        if not examples:
            raise ValueError("examples must not be empty")

        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        pad_token_id = int(eos_token_id) if eos_token_id is not None else 0

        self.inputs: list[torch.Tensor] = []
        self.labels: list[torch.Tensor] = []

        for messages in examples:
            token_ids, train_mask = chat_token_stream(messages, tokenizer)
            target_length = sequence_length + 1
            if len(token_ids) > target_length:
                token_ids = token_ids[-target_length:]
                train_mask = train_mask[-target_length:]

            if len(token_ids) < target_length:
                padding = target_length - len(token_ids)
                token_ids.extend([pad_token_id] * padding)
                train_mask.extend([False] * padding)

            inputs = torch.tensor(token_ids[:-1], dtype=torch.long)
            targets = torch.tensor(token_ids[1:], dtype=torch.long)
            target_mask = train_mask[1:]
            labels = targets.clone()
            labels[torch.tensor([not enabled for enabled in target_mask], dtype=torch.bool)] = IGNORE_INDEX

            if torch.all(labels == IGNORE_INDEX):
                raise ValueError("chat example contains no trainable assistant targets")

            self.inputs.append(inputs)
            self.labels.append(labels)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError("dataset index out of range")
        return {"input_ids": self.inputs[index], "labels": self.labels[index]}


def supervised_causal_language_model_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Compute causal LM loss while ignoring non-assistant targets."""
    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("logits must be (batch, sequence, vocab) and labels must be (batch, sequence)")
    if logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels sequence dimensions must match")

    trainable = labels.ne(ignore_index)
    if not torch.any(trainable):
        raise ValueError("labels contain no trainable targets")

    return nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=ignore_index,
    )


def sft_train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    grad_clip: float | None = None,
) -> float:
    """Run one supervised fine-tuning step."""
    if input_ids.ndim != 2 or labels.ndim != 2:
        raise ValueError("input_ids and labels must have shape (batch, sequence)")
    if input_ids.shape != labels.shape:
        raise ValueError("input_ids and labels must have matching shapes")
    if grad_clip is not None and grad_clip <= 0:
        raise ValueError("grad_clip must be positive when provided")

    model.train()
    logits = model(input_ids)
    loss = supervised_causal_language_model_loss(logits, labels)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if grad_clip is not None:
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return float(loss.detach())


def evaluate_sft(
    model: nn.Module,
    loader: object,
    device: torch.device | str = "cpu",
) -> float:
    """Evaluate supervised loss without updating parameters."""
    was_training = model.training
    model.eval()
    total_loss = 0.0
    batches = 0
    try:
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                total_loss += float(
                    supervised_causal_language_model_loss(model(input_ids), labels)
                )
                batches += 1
    finally:
        model.train(was_training)

    if batches == 0:
        raise ValueError("loader must contain at least one batch")
    return total_loss / batches
