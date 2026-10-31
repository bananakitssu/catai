import json

from catai.emotion_data import (
    daily_dialog_records,
    empathetic_dialogues_records,
    go_emotions_records,
)
from catai.emotion_merge import merge_emotion_datasets


def test_go_emotions_creates_multilabel_recognition_examples():
    records = go_emotions_records([
        {"text": "I am scared and furious.", "labels": [2, 14]},
    ])

    assert len(records) == 1
    assert records[0]["dataset_source"] == "go_emotions"
    assert records[0]["dataset_license"] == "apache-2.0"
    assert records[0]["emotion_labels"] == ["anger", "fear"]
    assert records[0]["messages"][-1]["content"] == "Emotions: anger, fear"


def test_daily_dialog_pairs_an_annotated_user_turn_with_real_reply():
    records = daily_dialog_records([
        {
            "dialog": ["I lost my wallet.", "That sounds stressful.", "I found it!"],
            "emotion": [5, 0, 4],
        }
    ])

    assert len(records) == 1
    assert records[0]["dataset_source"] == "daily_dialog"
    assert records[0]["emotion_labels"] == ["sadness"]
    assert records[0]["messages"][1]["content"] == "I lost my wallet."
    assert records[0]["messages"][2]["content"] == "That sounds stressful."
    assert "annotated as: sadness" in records[0]["messages"][0]["content"]


def test_empathetic_dialogues_uses_listener_replies_and_context():
    rows = [
        {
            "conv_id": "conversation-1",
            "utterance_idx": "1",
            "speaker_idx": "0",
            "context": "afraid",
            "prompt": "I was nervous about moving.",
            "utterance": "I was really scared to move away.",
        },
        {
            "conv_id": "conversation-1",
            "utterance_idx": "2",
            "speaker_idx": "1",
            "context": "afraid",
            "prompt": "I was nervous about moving.",
            "utterance": "That sounds like a big change.",
        },
        {
            "conv_id": "conversation-1",
            "utterance_idx": "3",
            "speaker_idx": "0",
            "context": "afraid",
            "prompt": "I was nervous about moving.",
            "utterance": "I did not know anyone there.",
        },
        {
            "conv_id": "conversation-1",
            "utterance_idx": "4",
            "speaker_idx": "1",
            "context": "afraid",
            "prompt": "I was nervous about moving.",
            "utterance": "It can be hard to start somewhere new.",
        },
    ]

    records = empathetic_dialogues_records(rows)

    assert len(records) == 2
    assert records[0]["dataset_source"] == "empathetic_dialogues"
    assert records[0]["emotion_labels"] == ["afraid"]
    assert records[0]["messages"][-1]["role"] == "assistant"
    assert records[1]["messages"][-1]["content"] == "It can be hard to start somewhere new."
    assert "annotated emotional context is: afraid" in records[0]["messages"][0]["content"]


def test_merge_emotion_datasets_retains_provenance_and_deduplicates(tmp_path):
    go = go_emotions_records([{"text": "I am angry.", "labels": [2]}])
    daily = daily_dialog_records([
        {"dialog": ["I am upset.", "What happened?"], "emotion": [5, 0]}
    ])
    empathetic = empathetic_dialogues_records([
        {
            "conv_id": "c1", "utterance_idx": "1", "speaker_idx": "0",
            "context": "sad", "prompt": "A difficult day.",
            "utterance": "Today was really difficult.",
        },
        {
            "conv_id": "c1", "utterance_idx": "2", "speaker_idx": "1",
            "context": "sad", "prompt": "A difficult day.",
            "utterance": "I'm sorry it was so difficult.",
        },
    ])

    paths = []
    for name, records in (("go", go), ("daily", daily), ("empathetic", empathetic)):
        path = tmp_path / f"{name}.jsonl"
        path.write_text(
            "\n".join(json.dumps(record) for record in records) + "\n",
            encoding="utf-8",
        )
        paths.append(path)

    output = tmp_path / "merged.jsonl"
    stats = merge_emotion_datasets(*paths, output, seed=42)

    assert stats["total"] == 3
    assert stats["duplicates_removed"] == 0
    merged = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert {record["dataset_source"] for record in merged} == {
        "go_emotions", "daily_dialog", "empathetic_dialogues"
    }
    assert all(record["dataset_license"] for record in merged)
