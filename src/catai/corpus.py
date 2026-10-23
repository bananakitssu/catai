"""Helpers for turning text corpora into CatAI token streams."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import torch

from .tokenizer import BPETokenizer, CharTokenizer, PRINTABLE_ASCII


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
    text = load_text(path, encoding=encoding)
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise TypeError("tokenizer must provide an encode method")
    tokens = _encode_prefix(tokenizer, text, max_tokens)
    if not tokens:
        raise ValueError("training corpus must contain at least one token")
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None and tokens[-1] != eos_token_id:
        tokens.append(eos_token_id)
    return tokens


def _encode_prefix(
    tokenizer: Tokenizer,
    text: str,
    max_tokens: int | None,
    *,
    chunk_chars: int = 65_536,
) -> list[int]:
    """Encode only the requested token prefix without materializing the full corpus."""
    if max_tokens is None:
        return list(tokenizer.encode(text))
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive when provided")

    if isinstance(tokenizer, CharTokenizer):
        # Each input character maps to exactly one token for this tokenizer.
        return list(tokenizer.encode(text[:max_tokens]))

    token_ids: list[int] = []
    start = 0
    while start < len(text) and len(token_ids) < max_tokens:
        end = min(start + chunk_chars, len(text))
        if end < len(text) and not text[end - 1].isspace():
            # Keep BPE merges stable by ending on whitespace when possible.
            limit = min(len(text), start + chunk_chars * 2)
            while end < limit and not text[end - 1].isspace():
                end += 1

        chunk_ids = tokenizer.encode(text[start:end])
        remaining = max_tokens - len(token_ids)
        token_ids.extend(chunk_ids[:remaining])
        start = end

    return token_ids


def tokenizer_and_tokens(
    path: str | Path,
    *,
    encoding: str = "utf-8",
    tokenizer_type: str = "char",
    vocab_size: int = 512,
    max_tokens: int | None = None,
    tokenizer_sample_chars: int = 250_000,
) -> tuple[Tokenizer, torch.Tensor]:
    """Build a tokenizer from a corpus and return its token tensor."""
    if tokenizer_sample_chars < 1:
        raise ValueError("tokenizer_sample_chars must be positive")

    text = load_text(path, encoding=encoding)
    if tokenizer_type == "char":
        tokenizer: Tokenizer = CharTokenizer.from_text(text)
    elif tokenizer_type == "bpe":
        # Learning merges across the full corpus is expensive. A fixed prefix
        # is enough for a first experiment; retain every corpus character.
        sample = text[:tokenizer_sample_chars]
        sample_characters = len(set(sample))
        corpus_characters = len(set(text))
        # Keep room for subword merges even if the sample contains many rare
        # Unicode characters. Those extra base characters are preserved above
        # the configured budget instead of consuming every merge slot.
        common_character_budget = min(
            sample_characters, len(set(PRINTABLE_ASCII))
        )
        merge_budget = max(0, vocab_size - common_character_budget - 2)
        effective_vocab_size = corpus_characters + merge_budget + 2
        tokenizer = BPETokenizer.from_text(
            sample,
            vocab_size=effective_vocab_size,
            extra_characters=text,
        )
    else:
        raise ValueError("tokenizer_type must be 'char' or 'bpe'")

    token_ids = _encode_prefix(tokenizer, text, max_tokens)
    if not token_ids:
        raise ValueError("training corpus must contain at least one token")
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is not None and token_ids[-1] != eos_token_id:
        token_ids.append(eos_token_id)
    return tokenizer, torch.tensor(token_ids, dtype=torch.long)
