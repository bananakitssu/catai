import json

from data import prepare_extra_sft_datasets as extra_sft_datasets
from data.prepare_extra_sft_datasets import (
    BLENDED_SKILL_TALK_SHA256,
    convert_blended_skill_talk_row,
    convert_gsm8k_row,
    convert_smol_smoltalk_row,
    convert_ultrachat_row,
)


def test_smol_smoltalk_uses_documented_messages_and_source_fields():
    item = convert_smol_smoltalk_row({
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hello!"},
        ],
        "source": "smol-magpie-ultra-short",
    })
    assert item is not None
    assert item["dataset_license"] == "Apache-2.0"
    assert item["upstream_source"] == "smol-magpie-ultra-short"
    assert item["messages"][-1] == {"role": "assistant", "content": "Hello!"}
    assert convert_smol_smoltalk_row({"source": "smol-magpie-ultra"}) is None


def test_blended_skill_talk_turns_and_context():
    item = convert_blended_skill_talk_row({
        "personas": [["free persona"], ["guided persona", "second line"]],
        "context_dataset": "wizard_of_wikipedia",
        "additional_context": "Astronomy",
        "free_turker_utterance": "I enjoy astronomy.",
        "guided_turker_utterance": "That is fascinating.",
        "dialog": [[0, "I also like the moon."], [1, "The moon has several phases."]],
    })
    assert item is not None
    assert item["dataset_license"] == "CC-BY-4.0"
    assert [message["role"] for message in item["messages"]] == [
        "system", "user", "assistant", "user", "assistant"
    ]
    assert "guided persona" in item["messages"][0]["content"]
    assert "Astronomy" in item["messages"][0]["content"]


def test_blended_skill_talk_checksum_matches_official_parlai_build():
    assert BLENDED_SKILL_TALK_SHA256 == (
        "5fbed0068ee89e2d43b93c3ecb341e784617033efa5e8e911a219d4eda6134a6"
    )



def test_smol_smoltalk_fetches_bounded_random_dataset_viewer_pages(monkeypatch, tmp_path):
    calls = []

    def fake_fetch_page(offset, length=100):
        calls.append((offset, length))
        total_rows = 350
        rows = [
            {
                "row_idx": index,
                "row": {
                    "messages": [
                        {"role": "user", "content": f"Question {index}"},
                        {"role": "assistant", "content": f"Answer {index}"},
                    ],
                    "source": "smol-magpie-ultra-short",
                },
            }
            for index in range(offset, min(offset + length, total_rows))
        ]
        return {"num_rows_total": total_rows, "partial": False, "rows": rows}

    monkeypatch.setattr(extra_sft_datasets, "_fetch_smol_smoltalk_page", fake_fetch_page)
    first_output = tmp_path / "first.jsonl"
    second_output = tmp_path / "second.jsonl"

    assert extra_sft_datasets.prepare_smol_smoltalk(first_output, limit=150, seed=42) == 150
    first_records = [json.loads(line) for line in first_output.read_text(encoding="utf-8").splitlines()]

    assert extra_sft_datasets.prepare_smol_smoltalk(second_output, limit=150, seed=42) == 150
    second_records = [json.loads(line) for line in second_output.read_text(encoding="utf-8").splitlines()]

    assert first_records == second_records
    assert len(first_records) == 150
    assert all(length <= 100 for _, length in calls)
    assert all(offset >= 0 for offset, _ in calls)


def test_smol_smoltalk_dataset_viewer_request_uses_default_train_split(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return json.dumps({
                "num_rows_total": 460341,
                "partial": False,
                "rows": [],
            }).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(extra_sft_datasets.urllib.request, "urlopen", fake_urlopen)
    payload = extra_sft_datasets._fetch_smol_smoltalk_page(200, 100)

    from urllib.parse import parse_qs, urlsplit

    query = parse_qs(urlsplit(captured["url"]).query)
    assert urlsplit(captured["url"]).path == "/rows"
    assert query == {
        "dataset": ["HuggingFaceTB/smol-smoltalk"],
        "config": ["default"],
        "split": ["train"],
        "offset": ["200"],
        "length": ["100"],
    }
    assert captured["timeout"] == 60
    assert payload["num_rows_total"] == 460341


def test_ultrachat_conversion_preserves_messages_and_mit_provenance():
    item = convert_ultrachat_row({
        "prompt_id": "prompt-123",
        "messages": [
            {"role": "user", "content": "Explain gravity simply."},
            {"role": "assistant", "content": "Gravity attracts objects with mass."},
        ],
    })
    assert item is not None
    assert item["dataset_license"] == "MIT"
    assert item["dataset_source"] == "ultrachat_200k"
    assert item["dataset_repository"] == "HuggingFaceH4/ultrachat_200k"
    assert item["messages"][-1]["content"] == "Gravity attracts objects with mass."
    assert convert_ultrachat_row({"messages": []}) is None


def test_gsm8k_conversion_preserves_published_answer_and_mit_provenance():
    item = convert_gsm8k_row({
        "question": "A box has 3 rows of 4 apples. How many apples?",
        "answer": "3 * 4 = 12.\\n#### 12",
    })
    assert item is not None
    assert item["dataset_license"] == "MIT"
    assert item["dataset_source"] == "gsm8k"
    assert item["dataset_repository"] == "openai/gsm8k"
    assert "3 rows of 4 apples" in item["messages"][0]["content"]
    assert item["messages"][1]["content"].endswith("#### 12")
    assert convert_gsm8k_row({"question": "No answer"}) is None


def test_dataset_viewer_fetch_uses_the_requested_repository_and_split(monkeypatch):
    from urllib.parse import parse_qs, urlsplit

    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return json.dumps({
                "num_rows_total": 10,
                "partial": False,
                "rows": [{"row_idx": 0, "row": {"question": "Q", "answer": "A"}}],
            }).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(extra_sft_datasets.urllib.request, "urlopen", fake_urlopen)
    payload = extra_sft_datasets._fetch_dataset_viewer_page(
        "openai/gsm8k", "main", "train", 0, 100
    )
    query = parse_qs(urlsplit(captured["url"]).query)
    assert query == {
        "dataset": ["openai/gsm8k"],
        "config": ["main"],
        "split": ["train"],
        "offset": ["0"],
        "length": ["100"],
    }
    assert captured["timeout"] == 60
    assert payload["rows"][0]["row"]["question"] == "Q"
