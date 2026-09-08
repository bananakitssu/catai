"""A deterministic character-level tokenizer with an optional end-of-sequence token."""

from __future__ import annotations

from dataclasses import dataclass


EOS_TOKEN = "<EOS>"


@dataclass(frozen=True)
class CharTokenizer:
    """Maps characters plus an optional special EOS token to integer token IDs."""

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
