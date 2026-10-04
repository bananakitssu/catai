"""Utilities for validating instruction and chat datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict


ALLOWED_ROLES = frozenset({"system", "user", "assistant"})
STATE_SECTIONS = {
    "emotions": (
        "happiness", "sadness", "affection", "curiosity", "excitement",
        "frustration", "anger", "fear", "calmness", "confidence",
        "loneliness", "playfulness",
    ),
    "needs": ("social_connection", "stimulation", "task_completion", "rest"),
    "personality": ("playful", "curious", "helpful", "affectionate", "seriousness"),
}


class ChatMessage(TypedDict):
    role: str
    content: str


def validate_state(state: object) -> dict[str, dict[str, float]]:
    """Validate CatAI's normalized 0..1 emotional/needs/personality state."""
    if not isinstance(state, dict):
        raise ValueError("state must be an object")

    normalized: dict[str, dict[str, float]] = {}
    for section, fields in STATE_SECTIONS.items():
        values = state.get(section)
        if not isinstance(values, dict):
            raise ValueError(f"state.{section} must be an object")
        section_values: dict[str, float] = {}
        for field in fields:
            value = values.get(field)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"state.{section}.{field} must be a number")
            value = float(value)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"state.{section}.{field} must be between 0 and 1")
            section_values[field] = value
        normalized[section] = section_values

    return normalized


def state_system_message(state: dict[str, dict[str, float]]) -> ChatMessage:
    """Render structured CatAI state as a deterministic system message."""
    return {
        "role": "system",
        "content": "CatAI internal state:\n" + json.dumps(
            state, separators=(",", ":"), ensure_ascii=True
        ),
    }


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
    if normalized[-1]["role"] != "assistant":
        raise ValueError("chat example must end with an assistant message")
    return normalized


def load_chat_dataset(path: str | Path, *, encoding: str = "utf-8") -> list[list[ChatMessage]]:
    """Load and validate a JSONL instruction/chat dataset.

    A record may contain a structured state object. When present, it is
    rendered into a system message so SFT learns to condition responses on
    CatAI's internal state without changing the chat message schema.
    """
    examples: list[list[ChatMessage]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding=encoding).splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record: Any = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {line_number}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"line {line_number} must contain a JSON object")
        messages = validate_messages(record.get("messages"))
        if "state" in record:
            state = validate_state(record["state"])
            messages = [state_system_message(state), *messages]
        examples.append(messages)

    if not examples:
        raise ValueError("chat dataset must contain at least one example")
    return examples
