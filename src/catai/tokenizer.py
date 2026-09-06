"""A deterministic character-level tokenizer for the first CatAI milestone."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CharTokenizer:
    """Maps a fixed vocabulary of characters to integer token IDs."""

    vocabulary: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.vocabulary:
            raise ValueError("vocabulary must not be empty")
        if len(set(self.vocabulary)) != len(self.vocabulary):
            raise ValueError("vocabulary contains duplicate tokens")

    @property
    def vocab_size(self) -> int:
        return len(self.vocabulary)

    def encode(self, text: str) -> list[int]:
        lookup = {token: i for i, token in enumerate(self.vocabulary)}
        unknown = sorted(set(text) - lookup.keys())
        if unknown:
            raise ValueError(f"unknown characters: {unknown!r}")
        return [lookup[char] for char in text]

    def decode(self, tokens: list[int]) -> str:
        if any(token < 0 or token >= self.vocab_size for token in tokens):
            raise ValueError("token ID outside vocabulary")
        return "".join(self.vocabulary[token] for token in tokens)

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        if not text:
            raise ValueError("training text must not be empty")
        return cls(tuple(sorted(set(text))))
