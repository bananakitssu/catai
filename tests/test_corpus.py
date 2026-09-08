from pathlib import Path

import pytest

from catai.corpus import load_text, load_tokens
from catai.tokenizer import CharTokenizer


def test_load_text_reads_utf8(tmp_path: Path) -> None:
    path = tmp_path / "corpus.txt"
    path.write_text("cat\nAI", encoding="utf-8")
    assert load_text(path) == "cat\nAI"


def test_load_tokens_encodes_text_and_appends_eos(tmp_path: Path) -> None:
    path = tmp_path / "corpus.txt"
    path.write_text("cat", encoding="utf-8")
    tokenizer = CharTokenizer.from_text("cat")
    assert load_tokens(path, tokenizer) == tokenizer.encode("cat") + [tokenizer.eos_token_id]


def test_load_tokens_preserves_line_breaks(tmp_path: Path) -> None:
    path = tmp_path / "corpus.txt"
    path.write_text("cat\nAI", encoding="utf-8")
    tokenizer = CharTokenizer.from_text("cat\nAI")
    tokens = load_tokens(path, tokenizer)
    assert tokenizer.decode(tokens) == "cat\nAI"
    assert tokens[-1] == tokenizer.eos_token_id


def test_load_tokens_requires_encode(tmp_path: Path) -> None:
    path = tmp_path / "corpus.txt"
    path.write_text("cat", encoding="utf-8")
    with pytest.raises(TypeError, match="encode"):
        load_tokens(path, object())
