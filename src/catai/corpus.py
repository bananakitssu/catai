"""Helpers for turning text corpora into CatAI token streams."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import torch

from .tokenizer import BPETokenizer, CharTokenizer


class Tokenizer(Protocol):
    """Minimal tokenizer interface required by the corpus loader."""

    vocab_size: int
    eos_token_id: int | None

    def encode(self, text: str) -> list[int]: ...

    def decode(self, tokens: list[int]) -> str: ...


def load_text(path: str | Path, *, encoding: str = "utf-8") -> str:
    """Read a text corpus from disk."""
    text = Path(path).read_text(encoding=encoding)
    if not text:
        raise ValueError("training corpus must not be empty")
    return text


def load_tokens(path: str | Path, tokenizer: Tokenizer, *, encoding: str = "utf-8", max_tokens: int | None = None) -> list[int]:
    """Load a corpus, encode it, append EOS, and optionally cap its token budget."""
    if max_tokens is not None and max_tokens < 1:
        raise ValueError("max_tokens must be positive when provided")
    text = load_text(path, encoding=encoding)
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise TypeError("tokenizer must provide an encode method")
    tokens = list(encode(text))
    if max_tokens is not None:
        tokens = tokens[:max_tokens]
    if not tokens:
        raise ValueError("training corpus must contain at least one token")
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None and (not tokens or tokens[-1] != eos_token_id):
        tokens.append(eos_token_id)
    return tokens


def tokenizer_and_tokens(
    path: str | Path,
    *,
    encoding: str = "utf-8",
    tokenizer_type: str = "char",
    vocab_size: int = 256,
    max_tokens: int | None = None,
) -> tuple[Tokenizer, torch.Tensor]:
    """Build a tokenizer from a corpus and return its token tensor."""
    text = load_text(path, encoding=encoding)
    if tokenizer_type == "char":
        tokenizer: Tokenizer = CharTokenizer.from_text(text)
    elif tokenizer_type == "bpe":
        tokenizer = BPETokenizer.from_text(text, vocab_size=vocab_size)
    else:
        raise ValueError("tokenizer_type must be 'char' or 'bpe'")
    tokens = torch.tensor(
        load_tokens(path, tokenizer, encoding=encoding, max_tokens=max_tokens),
        dtype=torch.long,
    )
    return tokenizer, tokens
