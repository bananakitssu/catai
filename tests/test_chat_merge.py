import json

import pytest

from catai.chat_dataset import DEFAULT_STATE
from catai.chat_merge import merge_chat_datasets


def _record(question: str, answer: str) -> dict:
    return {
        "state": DEFAULT_STATE,
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
    }


def _write(path, records):
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_merge_balances_sources_and_deduplicates_across_datasets(tmp_path):
    oasst1 = tmp_path / "oasst1.jsonl"
    oasst2 = tmp_path / "oasst2.jsonl"
    output = tmp_path / "merged.jsonl"
    _write(oasst1, [_record("Q1", "A1"), _record("Q2", "A2")])
    _write(oasst2, [_record("Q2", "A2"), _record("Q3", "A3")])

    stats = merge_chat_datasets(oasst1, oasst2, output, max_examples=3, seed=7)
    merged = _read(output)

    assert stats == {
        "oasst1_available": 2,
        "oasst2_available": 1,
        "duplicates_removed": 1,
        "oasst1_selected": 2,
        "oasst2_selected": 1,
        "total": 3,
    }
    assert {record["dataset_source"] for record in merged} == {"oasst1", "oasst2"}
    assert len({tuple((m["role"], m["content"]) for m in r["messages"]) for r in merged}) == 3


def test_merge_selection_is_deterministic_and_respects_total_cap(tmp_path):
    oasst1 = tmp_path / "oasst1.jsonl"
    oasst2 = tmp_path / "oasst2.jsonl"
    first_output = tmp_path / "first.jsonl"
    second_output = tmp_path / "second.jsonl"
    _write(oasst1, [_record(f"Q{i}", f"A{i}") for i in range(12)])
    _write(oasst2, [_record(f"R{i}", f"B{i}") for i in range(12)])

    stats1 = merge_chat_datasets(oasst1, oasst2, first_output, max_examples=7, seed=123)
    stats2 = merge_chat_datasets(oasst1, oasst2, second_output, max_examples=7, seed=123)

    assert stats1["total"] == 7
    assert stats1["oasst1_selected"] == 4
    assert stats1["oasst2_selected"] == 3
    assert first_output.read_bytes() == second_output.read_bytes()
    assert stats1 == stats2


def test_merge_fills_quota_from_source_that_has_more_examples(tmp_path):
    oasst1 = tmp_path / "oasst1.jsonl"
    oasst2 = tmp_path / "oasst2.jsonl"
    output = tmp_path / "merged.jsonl"
    _write(oasst1, [_record("Only Q", "Only A")])
    _write(oasst2, [_record(f"Q{i}", f"A{i}") for i in range(5)])

    stats = merge_chat_datasets(oasst1, oasst2, output, max_examples=5)
    assert stats["total"] == 5
    assert stats["oasst1_selected"] == 1
    assert stats["oasst2_selected"] == 4


def test_merge_rejects_malformed_dataset_record(tmp_path):
    oasst1 = tmp_path / "oasst1.jsonl"
    oasst2 = tmp_path / "oasst2.jsonl"
    _write(oasst1, [{"state": DEFAULT_STATE, "messages": [{"role": "user", "content": "Q"}]}])
    _write(oasst2, [_record("Q2", "A2")])

    with pytest.raises(ValueError, match="complete exchange"):
        merge_chat_datasets(oasst1, oasst2, tmp_path / "merged.jsonl")
