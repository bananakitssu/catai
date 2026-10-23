"""Tokenizers used by CatAI."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import heapq


EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"

# Fixed base vocabulary so the model can compose any string from known
# characters even if the full word never appeared in the training corpus.
# This is the key property that makes character-level useful for inventing
# names, codes, etc.
PRINTABLE_ASCII = (
    "\t\n\r"
    + " !\"#$%&'()*+,-./0123456789:;<=>?@"
    + "ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`"
    + "abcdefghijklmnopqrstuvwxyz{|}~"
)


def _fixed_char_vocabulary(extra: str = "") -> tuple[str, ...]:
    """Build a stable character vocabulary from printable ASCII + extras."""
    chars = set(PRINTABLE_ASCII)
    chars.update(extra)
    # Keep a deterministic order: printable ASCII first, then any extras, then specials.
    ordered = [c for c in PRINTABLE_ASCII if c in chars]
    extras = sorted(c for c in chars if c not in PRINTABLE_ASCII)
    return tuple(ordered + extras + [UNK_TOKEN, EOS_TOKEN])


@dataclass(frozen=True)
class CharTokenizer:
    """Maps characters plus special tokens to integer token IDs.

    Prefer the fixed printable-ASCII vocabulary so the model can always
    spell arbitrary names/codes from known characters.
    """

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
        return self.vocabulary.index(EOS_TOKEN) if EOS_TOKEN in self.vocabulary else None

    @property
    def unk_token_id(self) -> int | None:
        return self.vocabulary.index(UNK_TOKEN) if UNK_TOKEN in self.vocabulary else None

    def encode(self, text: str) -> list[int]:
        lookup = {token: i for i, token in enumerate(self.vocabulary)}
        unknown_id = self.unk_token_id
        unknown = sorted(set(text) - lookup.keys())
        if unknown and unknown_id is None:
            raise ValueError(f"unknown characters: {unknown!r}")
        return [lookup.get(char, unknown_id) for char in text]

    def decode(self, tokens: list[int]) -> str:
        if any(token < 0 or token >= self.vocab_size for token in tokens):
            raise ValueError("token ID outside vocabulary")
        decoded: list[str] = []
        for token in tokens:
            if token == self.eos_token_id:
                break
            if token == self.unk_token_id:
                decoded.append("�")
            else:
                decoded.append(self.vocabulary[token])
        return "".join(decoded)

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        """Build a tokenizer that always includes printable ASCII.

        Any additional characters present in *text* are also kept so the
        vocabulary is never smaller than the fixed base. This preserves the
        ability to invent unseen names while still covering the training data.
        """
        if not text:
            raise ValueError("training text must not be empty")
        return cls(_fixed_char_vocabulary(extra=text))

    @classmethod
    def default(cls) -> "CharTokenizer":
        """Return a tokenizer with only the fixed printable-ASCII vocabulary."""
        return cls(_fixed_char_vocabulary())


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
        return self.vocabulary.index(EOS_TOKEN) if EOS_TOKEN in self.vocabulary else None

    @property
    def unk_token_id(self) -> int | None:
        return self.vocabulary.index(UNK_TOKEN) if UNK_TOKEN in self.vocabulary else None

    def _apply_merges(self, symbols: list[str]) -> list[str]:
        """Apply ordered BPE merges without rescanning all text per merge."""
        if len(symbols) < 2 or not self.merges:
            return symbols

        ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        count = len(symbols)
        previous = [index - 1 for index in range(count)]
        following = [index + 1 for index in range(count)]
        following[-1] = -1
        alive = [True] * count
        candidates: list[tuple[int, int, int, tuple[str, str]]] = []

        def push_pair(left_index: int, right_index: int) -> None:
            if left_index < 0 or right_index < 0:
                return
            if not alive[left_index] or not alive[right_index]:
                return
            if following[left_index] != right_index:
                return
            pair = (symbols[left_index], symbols[right_index])
            rank = ranks.get(pair)
            if rank is not None:
                heapq.heappush(candidates, (rank, left_index, right_index, pair))

        for index in range(count - 1):
            push_pair(index, index + 1)

        while candidates:
            rank, left_index, right_index, pair = heapq.heappop(candidates)
            if (
                not alive[left_index]
                or not alive[right_index]
                or following[left_index] != right_index
                or (symbols[left_index], symbols[right_index]) != pair
                or ranks.get(pair) != rank
            ):
                continue

            symbols[left_index] += symbols[right_index]
            alive[right_index] = False
            after = following[right_index]
            following[left_index] = after
            if after >= 0:
                previous[after] = left_index
            push_pair(previous[left_index], left_index)
            push_pair(left_index, after)

        merged: list[str] = []
        index = 0
        while index >= 0:
            if alive[index]:
                merged.append(symbols[index])
            index = following[index]
        return merged

    def encode(self, text: str) -> list[int]:
        lookup = {token: i for i, token in enumerate(self.vocabulary)}
        unknown_id = self.unk_token_id
        symbols = self._apply_merges(list(text))
        result: list[int] = []
        for symbol in symbols:
            token_id = lookup.get(symbol)
            if token_id is None:
                if unknown_id is None:
                    raise ValueError(f"unknown text pieces: {[symbol]!r}")
                token_id = unknown_id
            result.append(token_id)
        return result

    def decode(self, tokens: list[int]) -> str:
        if any(token < 0 or token >= self.vocab_size for token in tokens):
            raise ValueError("token ID outside vocabulary")
        decoded: list[str] = []
        for token in tokens:
            if token == self.eos_token_id:
                break
            if token == self.unk_token_id:
                decoded.append("�")
            else:
                decoded.append(self.vocabulary[token])
        return "".join(decoded)

    @classmethod
    def from_text(
        cls,
        text: str,
        vocab_size: int = 256,
        *,
        extra_characters: str = "",
    ) -> "BPETokenizer":
        """Train BPE merges on a sample while optionally retaining corpus characters."""
        if not text:
            raise ValueError("training text must not be empty")
        if vocab_size < 3:
            raise ValueError("vocab_size must be at least 3")

        # BPE starts with the characters that actually occur in the corpus.
        # Unlike the character tokenizer, it must honor the requested vocabulary
        # budget so small vocabularies remain useful and deterministic.
        symbols = list(text)
        base_vocabulary = set(symbols)
        base_vocabulary.update(extra_characters)
        special_count = 2
        if len(base_vocabulary) + special_count > vocab_size:
            raise ValueError(
                "vocab_size is too small to represent all corpus characters "
                "plus UNK/EOS tokens"
            )

        vocabulary = set(base_vocabulary)
        merges: list[tuple[str, str]] = []
        target = vocab_size - special_count

        while len(vocabulary) < target:
            # Do not learn merges across whitespace boundaries. This keeps
            # subword tokens meaningful and prevents a merge from changing
            # how neighboring words are segmented during encoding.
            counts = Counter()
            for left, right in zip(symbols, symbols[1:]):
                if left.isspace() or right.isspace():
                    continue
                counts[(left, right)] += 1

            candidates = [pair for pair in counts if pair[0] + pair[1] not in vocabulary]
            if not candidates:
                break

            best = max(candidates, key=lambda pair: (counts[pair], pair))
            left, right = best
            merged_token = left + right
            rewritten: list[str] = []
            index = 0
            while index < len(symbols):
                if (
                    index + 1 < len(symbols)
                    and symbols[index] == left
                    and symbols[index + 1] == right
                    and not left.isspace()
                    and not right.isspace()
                ):
                    rewritten.append(merged_token)
                    index += 2
                else:
                    rewritten.append(symbols[index])
                    index += 1
            symbols = rewritten
            vocabulary.add(merged_token)
            merges.append(best)

        return cls(tuple(sorted(vocabulary)) + (UNK_TOKEN, EOS_TOKEN), tuple(merges))
