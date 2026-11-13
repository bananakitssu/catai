"""Helpers for turning text corpora into CatAI token streams."""

from __future__ import annotations

from array import array
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


def _read_tokenizer_inputs(
    path: str | Path,
    *,
    encoding: str,
    sample_chars: int,
    scan_chunk_chars: int = 1_048_576,
) -> tuple[str, set[str]]:
    """Read a small tokenizer sample and collect unique characters by streaming."""
    if sample_chars < 1:
        raise ValueError("tokenizer_sample_chars must be positive")
    if scan_chunk_chars < 1:
        raise ValueError("scan_chunk_chars must be positive")

    characters: set[str] = set()
    with Path(path).open("r", encoding=encoding) as source:
        sample = source.read(sample_chars)
        if not sample:
            raise ValueError("training corpus must not be empty")
        characters.update(sample)
        while chunk := source.read(scan_chunk_chars):
            characters.update(chunk)
    return sample, characters


def _encode_prefix_tensor_from_file(
    tokenizer: Tokenizer,
    path: str | Path,
    max_tokens: int | None,
    *,
    encoding: str = "utf-8",
    chunk_chars: int = 65_536,
) -> torch.Tensor:
    """Encode a text file in bounded chunks without loading it all into memory."""
    if max_tokens is not None and max_tokens < 1:
        raise ValueError("max_tokens must be positive when provided")
    if chunk_chars < 1:
        raise ValueError("chunk_chars must be positive")

    token_ids = array("q")
    if token_ids.itemsize != 8:
        raise RuntimeError("the platform does not provide an 8-byte signed array type")

    with Path(path).open("r", encoding=encoding) as source:
        while max_tokens is None or len(token_ids) < max_tokens:
            chunk = source.read(chunk_chars)
            if not chunk:
                break

            # BPE merges never cross whitespace. Extend a partial final word
            # so neither it nor a structural <EOS> marker is split between chunks.
            if not chunk[-1].isspace():
                tail: list[str] = []
                while True:
                    character = source.read(1)
                    if not character:
                        break
                    tail.append(character)
                    if character.isspace():
                        break
                if tail:
                    chunk += "".join(tail)

            chunk_ids = tokenizer.encode(chunk)
            if max_tokens is None:
                token_ids.extend(chunk_ids)
            else:
                remaining = max_tokens - len(token_ids)
                token_ids.extend(chunk_ids[:remaining])

    if not token_ids:
        raise ValueError("training corpus must contain at least one token")
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is not None and token_ids[-1] != eos_token_id:
        token_ids.append(eos_token_id)

    # A contiguous native int64 buffer lets PyTorch view the IDs without
    # first allocating a Python int for every token or making another copy.
    return torch.frombuffer(token_ids, dtype=torch.int64)


def tokenizer_and_tokens(
    path: str | Path,
    *,
    encoding: str = "utf-8",
    tokenizer_type: str = "char",
    vocab_size: int = 512,
    max_tokens: int | None = None,
    tokenizer_sample_chars: int = 250_000,
    tokenizer: Tokenizer | None = None,
) -> tuple[Tokenizer, torch.Tensor]:
    """Build a tokenizer from a corpus and return its token tensor."""
    if tokenizer_sample_chars < 1:
        raise ValueError("tokenizer_sample_chars must be positive")

    if tokenizer_type not in {"char", "bpe"}:
        raise ValueError("tokenizer_type must be 'char' or 'bpe'")

    if tokenizer is None:
        sample, characters = _read_tokenizer_inputs(
            path,
            encoding=encoding,
            sample_chars=tokenizer_sample_chars,
        )
        if tokenizer_type == "char":
            tokenizer = CharTokenizer.from_characters(characters)
        else:
            # Learn merges from a bounded prefix while retaining the observed
            # character set without materializing the entire corpus as one string.
            tokenizer = BPETokenizer.from_text(
                sample,
                vocab_size=vocab_size,
                extra_characters=characters,
            )
    elif (
        (tokenizer_type == "char" and not isinstance(tokenizer, CharTokenizer))
        or (tokenizer_type == "bpe" and not isinstance(tokenizer, BPETokenizer))
    ):
        raise ValueError("provided tokenizer type does not match tokenizer_type")

    tokens = _encode_prefix_tensor_from_file(
        tokenizer, path, max_tokens, encoding=encoding
    )
    return tokenizer, tokens
