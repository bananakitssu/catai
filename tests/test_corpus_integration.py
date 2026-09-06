from pathlib import Path

import torch

from catai.corpus import tokenizer_and_tokens


def test_tiny_corpus_round_trips() -> None:
    path = Path(__file__).parents[1] / "data" / "tiny.txt"
    tokenizer, tokens = tokenizer_and_tokens(path)

    assert isinstance(tokens, torch.Tensor)
    assert tokens.dtype == torch.long
    assert tokenizer.decode(tokens.tolist()) == path.read_text(encoding="utf-8")
    assert tokens.numel() > 100
