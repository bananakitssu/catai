"""Helpers for turning text corpora into CatAI token streams."""

from __future__ import annotations

from pathlib import Path

import torch

from .tokenizer import CharTokenizer


def load_text(path: str | Path, *, encoding: str = "utf-8") -> str:
    """Read a text corpus from disk."""
    text = Path(path).read_text(encoding=encoding)
    if not text:
        raise ValueError("training corpus must not be empty")
    return text


def load_tokens(path: str | Path, tokenizer: CharTokenizer, *, encoding: str = "utf-8") -> torch.Tensor:
    """Load a text corpus and encode it as a tensor of token IDs."""
    return torch.tensor(tokenizer.encode(load_text(path, encoding=encoding)), dtype=torch.long)


def tokenizer_and_tokens(path: str | Path, *, encoding: str = "utf-8") -> tuple[CharTokenizer, torch.Tensor]:
    """Build a character tokenizer from a corpus and return its token stream."""
    tokenizer = CharTokenizer.from_text(load_text(path, encoding=encoding))
    return tokenizer, load_tokens(path, tokenizer, encoding=encoding)
