"""Helpers for turning text corpora into CatAI token streams."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import torch

from .tokenizer import CharTokenizer


class Tokenizer(Protocol):
    """Minimal tokenizer interface required by the corpus loader."""

    def encode(self, text: str) -> list[int]: ...



def load_text(path: str | Path, *, encoding: str = "utf-8") -> str:
    """Read a text corpus from disk."""
    text = Path(path).read_text(encoding=encoding)
    if not text:
        raise ValueError("training corpus must not be empty")
    return text


def load_tokens(path: str | Path, tokenizer: CharTokenizer, *, encoding: str = "utf-8") -> list[int]:
    """Load a text corpus and append EOS after every non-empty line."""
    text = load_text(path, encoding=encoding)
    tokens: list[int] = []
    for line in text.splitlines():
        if line:
            tokens.extend(tokenizer.encode(line))
            tokens.append(tokenizer.eos_token_id)
    if not tokens:
        raise ValueError("training corpus must contain at least one non-empty line")
    return tokens


def tokenizer_and_tokens(path: str | Path, *, encoding: str = "utf-8") -> tuple[CharTokenizer, torch.Tensor]:
    """Build a character tokenizer from a corpus and return its token tensor."""
    text = load_text(path, encoding=encoding)
    tokenizer = CharTokenizer.from_text(text)
    tokens = torch.tensor(load_tokens(path, tokenizer, encoding=encoding), dtype=torch.long)
    return tokenizer, tokens
