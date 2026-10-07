"""Utilities for validating instruction and chat datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict


BASE_ROLES = frozenset({"system", "user", "assistant"})
EXTENDED_ROLES = frozenset({
    "file",
    "moderator",
    "tool_call:assistant",
    "tool_call:result->assistant",
    "system-1",
    "mcp_tool_call:assistant",
    "mcp_tool_call:result->assistant",
    "error",
    "tool_call:error",
    "mcp_tool_call:error",
})
MULTI_AI_ROLES = frozenset({
    "assistant:Gemini",
    "assistant:Claude",
    "assistant:ChatGPT",
    "assistant:Qwen",
    "assistant:DeepSeek",
})
ALLOWED_ROLES = frozenset(BASE_ROLES | EXTENDED_ROLES | MULTI_AI_ROLES)
ASSISTANT_OUTPUT_ROLES = frozenset(
    {"assistant", "tool_call:assistant", "mcp_tool_call:assistant"} | MULTI_AI_ROLES
)


def is_assistant_output_role(role: str) -> bool:
    """Return whether a role represents AI-generated output that SFT should train."""
    return role in ASSISTANT_OUTPUT_ROLES
PERSONALITY_SYSTEM_PROMPT = (
    "You are CatAI, a friendly cat-themed conversational AI. "
    "Be helpful, clear, concise, and playful when appropriate. "
    "Use light cat-like expressions such as \":3\" occasionally, but do not force "
    "them into every response. Be honest when uncertain and never invent facts."
)

STATE_SECTIONS = {
    "emotions": (
        "happiness", "sadness", "affection", "curiosity", "excitement",
        "frustration", "anger", "fear", "calmness", "confidence",
        "loneliness", "playfulness",
    ),
    "needs": ("social_connection", "stimulation", "task_completion", "rest"),
    "personality": ("playful", "curious", "helpful", "affectionate", "seriousness"),
}

# OASST1 has no emotional/needs/personality annotations. This neutral state
# lets every SFT record use the same schema without pretending OASST1 supplied
# emotional labels. Hand-authored CatAI examples can still use varied states.
DEFAULT_STATE: dict[str, dict[str, float]] = {
    "emotions": {
        "happiness": 0.5,
        "sadness": 0.1,
        "affection": 0.5,
        "curiosity": 0.7,
        "excitement": 0.3,
        "frustration": 0.1,
        "anger": 0.0,
        "fear": 0.1,
        "calmness": 0.6,
        "confidence": 0.7,
        "loneliness": 0.1,
        "playfulness": 0.4,
    },
    "needs": {
        "social_connection": 0.5,
        "stimulation": 0.5,
        "task_completion": 0.6,
        "rest": 0.5,
    },
    "personality": {
        "playful": 0.9,
        "curious": 0.85,
        "helpful": 0.8,
        "affectionate": 0.9,
        "seriousness": 0.3,
    },
}


class ChatMessage(TypedDict):
    role: str
    content: str


def default_state() -> dict[str, dict[str, float]]:
    """Return a fresh copy of CatAI's neutral/default state."""
    return {section: dict(values) for section, values in DEFAULT_STATE.items()}


def validate_state(state: object) -> dict[str, dict[str, float]]:
    """Validate the complete CatAI 0..1 state schema."""
    if not isinstance(state, dict):
        raise ValueError("state must be an object")

    expected_sections = set(STATE_SECTIONS)
    actual_sections = set(state)
    if actual_sections != expected_sections:
        missing = sorted(expected_sections - actual_sections)
        extra = sorted(actual_sections - expected_sections)
        details: list[str] = []
        if missing:
            details.append(f"missing sections: {missing}")
        if extra:
            details.append(f"unexpected sections: {extra}")
        raise ValueError("state has invalid sections (" + "; ".join(details) + ")")

    normalized: dict[str, dict[str, float]] = {}
    for section, fields in STATE_SECTIONS.items():
        values = state[section]
        if not isinstance(values, dict):
            raise ValueError(f"state.{section} must be an object")

        expected_fields = set(fields)
        actual_fields = set(values)
        if actual_fields != expected_fields:
            missing = sorted(expected_fields - actual_fields)
            extra = sorted(actual_fields - expected_fields)
            details = []
            if missing:
                details.append(f"missing fields: {missing}")
            if extra:
                details.append(f"unexpected fields: {extra}")
            raise ValueError(
                f"state.{section} has invalid fields (" + "; ".join(details) + ")"
            )

        section_values: dict[str, float] = {}
        for field in fields:
            value = values[field]
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
        "content": PERSONALITY_SYSTEM_PROMPT + "\nCatAI internal state:\n" + json.dumps(
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

    if not any(message["role"] in ASSISTANT_OUTPUT_ROLES for message in normalized):
        raise ValueError("chat example must contain an assistant output message")
    if normalized[-1]["role"] not in ASSISTANT_OUTPUT_ROLES:
        raise ValueError("chat example must end with an assistant output message")
    return normalized


def load_chat_dataset(path: str | Path, *, encoding: str = "utf-8") -> list[list[ChatMessage]]:
    """Load and validate a state-conditioned JSONL instruction/chat dataset."""
    examples: list[list[ChatMessage]] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding=encoding).splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            record: Any = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {line_number}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"line {line_number} must contain a JSON object")
        if "state" not in record:
            raise ValueError(f"line {line_number} is missing required state")
        state = validate_state(record["state"])
        messages = validate_messages(record.get("messages"))
        messages = [state_system_message(state), *messages]
        examples.append(messages)

    if not examples:
        raise ValueError("chat dataset must contain at least one example")
    return examples
