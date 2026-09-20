"""Utilities for validating instruction and chat datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict


ALLOWED_ROLES = frozenset({"system", "user", "assistant"})


class ChatMessage(TypedDict):
    role: str
    content: str


def validate_messages(messages: object) -> list[ChatMessage]:
    """Validate and normalize one chat example's messages."""
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")

    normalized: list[ChatMessage] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("each message must be an object")
        role = message.get("role")
        content = message.get("content")
        if role not in ALLOWED_ROLES:
            raise ValueError(f"unsupported message role: {role!r}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("message content must be a non-empty string")
        normalized.append({"role": role, "content": content})

    if not any(message["role"] == "assistant" for message in normalized):
        raise ValueError("chat example must contain an assistant message")
    return normalized


def load_chat_dataset(path: str | Path, *, encoding: str = "utf-8") -> list[list[ChatMessage]]:
    """Load and validate a JSONL instruction/chat dataset."""
    examples: list[list[ChatMessage]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding=encoding).splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {line_number}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"line {line_number} must contain a JSON object")
        messages = validate_messages(record.get("messages"))
        examples.append(messages)

    if not examples:
        raise ValueError("chat dataset must contain at least one example")
    return examples
