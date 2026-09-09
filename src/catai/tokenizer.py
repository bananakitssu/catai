"""Tokenizers used by CatAI."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


EOS_TOKEN = "<EOS>"


@dataclass(frozen=True)
class CharTokenizer:
    """Maps characters plus an optional EOS token to integer token IDs."""

    vocabulary: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.vocabulary:
            raise ValueError("vocabulary must not be empty")
        if len(set(self.vocabulary)) != len(self.vocabulary):
            raise ValueError("vocabulary contains duplicate tokens")
        if EOS_TOKEN in self.vocabulary and self.vocabulary[-1] != EOS_TOKEN:
            raise ValueError(f"{EOS_TOKEN} must be the final vocabulary token")

    @property
    def vocab_size(self) -> int:
        return len(self.vocabulary)

    @property
    def eos_token_id(self) -> int | None:
        if EOS_TOKEN not in self.vocabulary:
            return None
        return self.vocabulary.index(EOS_TOKEN)

    def encode(self, text: str) -> list[int]:
        lookup = {token: i for i, token in enumerate(self.vocabulary)}
        unknown = sorted(set(text) - lookup.keys())
        if unknown:
            raise ValueError(f"unknown characters: {unknown!r}")
        return [lookup[char] for char in text]

    def decode(self, tokens: list[int]) -> str:
        if any(token < 0 or token >= self.vocab_size for token in tokens):
            raise ValueError("token ID outside vocabulary")
        decoded: list[str] = []
        for token in tokens:
            if self.eos_token_id is not None and token == self.eos_token_id:
                break
            decoded.append(self.vocabulary[token])
        return "".join(decoded)

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        if not text:
            raise ValueError("training text must not be empty")
        return cls(tuple(sorted(set(text))) + (EOS_TOKEN,))


@dataclass(frozen=True)
class BPETokenizer:
    """A deterministic byte-pair-style subword tokenizer trained from text."""

    vocabulary: tuple[str, ...]
    merges: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.vocabulary:
            raise ValueError("vocabulary must not be empty")
        if len(set(self.vocabulary)) != len(self.vocabulary):
            raise ValueError("vocabulary contains duplicate tokens")
        if EOS_TOKEN in self.vocabulary and self.vocabulary[-1] != EOS_TOKEN:
            raise ValueError(f"{EOS_TOKEN} must be the final vocabulary token")
        vocab = set(self.vocabulary)
        for left, right in self.merges:
            if not left or not right or left + right not in vocab:
                raise ValueError("merges must reference tokens present in the vocabulary")

    @property
    def vocab_size(self) -> int:
        return len(self.vocabulary)

    @property
    def eos_token_id(self) -> int | None:
        if EOS_TOKEN not in self.vocabulary:
            return None
        return self.vocabulary.index(EOS_TOKEN)

    def _apply_merges(self, symbols: list[str]) -> list[str]:
        for left, right in self.merges:
            merged: list[str] = []
            index = 0
            while index < len(symbols):
                if index + 1 < len(symbols) and symbols[index] == left and symbols[index + 1] == right:
                    merged.append(left + right)
                    index += 2
                else:
                    merged.append(symbols[index])
                    index += 1
            symbols = merged
        return symbols

    def encode(self, text: str) -> list[int]:
        symbols = self._apply_merges(list(text))
        lookup = {token: i for i, token in enumerate(self.vocabulary)}
        unknown = [token for token in symbols if token not in lookup]
        if unknown:
            raise ValueError(f"unknown text pieces: {sorted(set(unknown))!r}")
        return [lookup[token] for token in symbols]

    def decode(self, tokens: list[int]) -> str:
        if any(token < 0 or token >= self.vocab_size for token in tokens):
            raise ValueError("token ID outside vocabulary")
        decoded: list[str] = []
        for token in tokens:
            if self.eos_token_id is not None and token == self.eos_token_id:
                break
            decoded.append(self.vocabulary[token])
        return "".join(decoded)

    @classmethod
    def from_text(cls, text: str, vocab_size: int = 256) -> "BPETokenizer":
        """Train a small deterministic BPE vocabulary from a text sample."""
        if not text:
            raise ValueError("training text must not be empty")
        if vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")

        symbols = list(text)
        base_vocabulary = set(symbols)
        if len(base_vocabulary) + 1 >= vocab_size:
            vocabulary = tuple(sorted(base_vocabulary)) + (EOS_TOKEN,)
            return cls(vocabulary)

        vocabulary = set(base_vocabulary)
        merges: list[tuple[str, str]] = []
        target = vocab_size - 1
        while len(vocabulary) < target:
            counts = Counter(zip(symbols, symbols[1:]))
            candidates = [pair for pair in counts if pair[0] + pair[1] not in vocabulary]
            if not candidates:
                break
            best = max(candidates, key=lambda pair: (counts[pair], pair))
            left, right = best
            merged_token = left + right
            rewritten: list[str] = []
            index = 0
            while index < len(symbols):
                if index + 1 < len(symbols) and symbols[index] == left and symbols[index + 1] == right:
                    rewritten.append(merged_token)
                    index += 2
                else:
                    rewritten.append(symbols[index])
                    index += 1
            symbols = rewritten
            vocabulary.add(merged_token)
            merges.append(best)

        return cls(tuple(sorted(vocabulary)) + (EOS_TOKEN,), tuple(merges))
