"""Helpers for loading text corpora."""

from __future__ import annotations

from pathlib import Path


def load_text(path: str | Path, *, encoding: str = "utf-8") -> str:
    """Read a UTF-8 text corpus from disk."""
    return Path(path).read_text(encoding=encoding)


def load_tokens(path: str | Path, tokenizer: object, *, encoding: str = "utf-8") -> list[int]:
    """Load a text corpus and encode it with a tokenizer."""
    text = load_text(path, encoding=encoding)
    encode = getattr(tokenizer, "encode", None)
    if encode is None:
        raise TypeError("tokenizer must provide an encode method")
    return list(encode(text))
