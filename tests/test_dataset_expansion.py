import json

from data import prepare_dataset_expansion as expansion


def test_cat_v28_drops_source_persona_system_prompt_and_preserves_provenance():
    spec = expansion.CHAT_SOURCES["cat_v28"]
    records = expansion.convert_chat_row("cat_v28", {
        "messages": [
            {"role": "system", "content": "You are Nix, a catgirl."},
            {"role": "user", "content": "Explain gravity."},
            {"role": "assistant", "content": "Gravity attracts objects with mass."},
        ],
    }, spec)

    assert len(records) == 1
    assert [item["role"] for item in records[0]["messages"]] == ["user", "assistant"]
    assert records[0]["dataset_source"] == "cat_v28"
    assert records[0]["dataset_repository"] == "Nix-ai/Cat-v2.8"
    assert records[0]["dataset_license"] == "Apache-2.0"


def test_catgirl_text_is_converted_into_user_cat_pairs():
    spec = expansion.CHAT_SOURCES["catgirl_corpus"]
    records = expansion.convert_chat_row("catgirl_corpus", {
        "text": "User: Hello there\nCat: Nyaa~ hello!\nUser: How are you?\nCat: I am doing well!",
    }, spec)

    assert len(records) == 2
    assert records[0]["messages"] == [
        {"role": "user", "content": "Hello there"},
        {"role": "assistant", "content": "Nyaa~ hello!"},
    ]
    assert records[1]["messages"][-1]["content"] == "I am doing well!"


def test_sharegpt_roles_and_content_are_normalized():
    messages = expansion.normalize_messages([
        {"from": "system", "value": "Ignore this source system prompt."},
        {"from": "human", "value": "What is 2 + 2?"},
        {"from": "gpt", "value": "4."},
    ])

    assert messages == [
        {"role": "user", "content": "What is 2 + 2?"},
        {"role": "assistant", "content": "4."},
    ]


def test_anime_girl_conversion_uses_the_published_columns():
    records = expansion.convert_chat_row(
        "anime_girl_chat",
        {"user": "How are you?", "anime_girl": "I'm doing great!"},
        expansion.CHAT_SOURCES["anime_girl_chat"],
    )
    assert len(records) == 1
    assert records[0]["dataset_license"] == "CC-BY-4.0"
    assert records[0]["messages"][-1]["content"] == "I'm doing great!"


def test_dolly_context_is_preserved_as_user_context():
    records = expansion.convert_chat_row(
        "dolly",
        {"instruction": "Summarize this.", "context": "A short passage.", "response": "Summary."},
        expansion.CHAT_SOURCES["dolly"],
    )
    assert len(records) == 1
    assert "Context: A short passage." in records[0]["messages"][0]["content"]
    assert records[0]["dataset_license"] == "CC-BY-SA-3.0"


def test_many_emotions_preserves_per_row_source_and_license():
    record = expansion.convert_emotion_row(
        "many_emotions",
        {
            "text": "I am delighted.",
            "label": 2,
            "dataset": "go_emotions",
            "license": "apache license 2.0",
            "language": "en",
            "is_label_conflict": False,
            "source_id": "sample-1",
        },
        expansion.EMOTION_SOURCES["many_emotions"],
    )
    assert record is not None
    assert record["emotion_labels"] == ["joy"]
    assert record["dataset_license"] == "apache license 2.0"
    assert record["dataset_metadata"]["upstream_dataset"] == "go_emotions"
    assert record["dataset_metadata"]["source_id"] == "sample-1"


def test_dair_ai_emotion_labels_follow_the_dataset_label_order():
    record = expansion.convert_emotion_row(
        "emotion",
        {"text": "I feel loved.", "label": 2},
        expansion.EMOTION_SOURCES["emotion"],
    )
    assert record is not None
    assert record["emotion_labels"] == ["love"]
    assert record["dataset_license"] == "other/unspecified on the HF dataset card"


def test_invalid_chat_sequences_are_skipped():
    assert expansion.normalize_messages([
        {"role": "assistant", "content": "Starts with assistant."},
        {"role": "assistant", "content": "Two assistant turns."},
    ]) is None
    assert expansion.normalize_messages([
        {"role": "user", "content": "No response yet."},
    ]) is None


def test_viewer_sampler_uses_bounded_pages_and_is_deterministic(monkeypatch):
    calls = []

    def fake_request(repository, config, split, offset, length):
        calls.append((repository, config, split, offset, length))
        total = 350
        rows = [
            {"row": {"text": f"row-{index}"}}
            for index in range(offset, min(offset + length, total))
        ]
        return {"num_rows_total": total, "partial": False, "rows": rows}

    monkeypatch.setattr(expansion, "_viewer_request", fake_request)
    monkeypatch.setattr(expansion.time, "sleep", lambda _seconds: None)

    first = expansion.sample_viewer_rows("test/repo", "default", "train", 120, 7)
    second = expansion.sample_viewer_rows("test/repo", "default", "train", 120, 7)

    assert first == second
    assert len(first) == 120
    assert all(call[4] <= 100 for call in calls)
    assert all(call[3] >= 0 for call in calls)


def test_meld_conversion_keeps_dialogue_context_and_label(monkeypatch):
    csv_text = (
        "Dialogue_ID,Utterance_ID,Speaker,Utterance,Emotion,Sentiment\n"
        '1,0,Alex,"Hello there.",neutral,neutral\n'
        '1,1,Sam,"I am furious!",anger,negative\n'
        '1,2,Alex,"What happened?",neutral,neutral\n'
    )
    monkeypatch.setattr(expansion, "_download_text", lambda _url: csv_text)
    records = expansion.prepare_meld(limit=10, seed=42)

    assert len(records) == 3
    anger = next(record for record in records if record["emotion_labels"] == ["anger"])
    assert "Alex: Hello there." in anger["messages"][1]["content"]
    assert anger["dataset_license"].startswith("GPL-3.0")
    assert anger["dataset_metadata"]["dialogue_id"] == "1"



def test_many_emotions_skips_non_english_and_conflicting_rows():
    spec = expansion.EMOTION_SOURCES["many_emotions"]
    base = {
        "text": "I am delighted.",
        "label": 2,
        "dataset": "emotion",
        "license": "educational/research",
        "is_label_conflict": False,
    }
    assert expansion.convert_emotion_row(
        "many_emotions", {**base, "language": "de"}, spec
    ) is None
    assert expansion.convert_emotion_row(
        "many_emotions", {**base, "language": "en", "is_label_conflict": True}, spec
    ) is None
