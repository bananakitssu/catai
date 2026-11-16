from __future__ import annotations

import json

import pytest

from catai.chat_dataset import load_chat_dataset
from data.prepare_markdown_sft import FORMATS, build_examples, prepare


def test_markdown_sft_covers_the_requested_markdown_features(tmp_path):
    output = tmp_path / "markdown_sft_chat.jsonl"

    count = prepare(output, max_examples=72, seed=42)

    assert count == 72
    raw_records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(raw_records) == 72
    assert {record["dataset_category"] for record in raw_records} == set(FORMATS)
    assert all(record["dataset_source"] == "catai_markdown_sft" for record in raw_records)
    assert all(record["dataset_license"] == "Apache-2.0" for record in raw_records)

    assistant_text = "\n".join(record["messages"][-1]["content"] for record in raw_records)
    assert "# " in assistant_text
    assert "- " in assistant_text
    assert "1. " in assistant_text
    assert "| --- |" in assistant_text
    assert "- [ ] " in assistant_text
    assert chr(96) * 3 + "python" in assistant_text
    assert "](https://" in assistant_text
    assert "> **Important:**" in assistant_text
    assert "\n---\n" in assistant_text

    loaded = load_chat_dataset(output)
    assert len(loaded) == 72
    assert all(example[-1]["role"] == "assistant" for example in loaded)


def test_markdown_sft_is_deterministic_and_enforces_limits(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"

    prepare(first, max_examples=24, seed=7)
    prepare(second, max_examples=24, seed=7)

    assert first.read_bytes() == second.read_bytes()
    assert len(build_examples()) == 72

    with pytest.raises(ValueError, match="cannot exceed"):
        prepare(tmp_path / "too-many.jsonl", max_examples=73)
