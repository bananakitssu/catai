import json

from data import prepare_first_wave_datasets as first_wave


def test_soda_dialogue_maps_two_speakers_to_user_and_assistant():
    messages = first_wave.convert_soda_row({
        "dialogue": ["Hello there.", "Hi! How can I help?", "What is HTTP?", "I can explain that."],
        "speakers": ["Alex", "Sam", "Alex", "Sam"],
    })

    assert messages == [
        {"role": "user", "content": "Hello there."},
        {"role": "assistant", "content": "Hi! How can I help?"},
        {"role": "user", "content": "What is HTTP?"},
        {"role": "assistant", "content": "I can explain that."},
    ]


def test_soda_rejects_misaligned_or_multi_party_dialogues():
    assert first_wave.convert_soda_row({
        "dialogue": ["Hi", "Hello"], "speakers": ["A"]
    }) is None
    assert first_wave.convert_soda_row({
        "dialogue": ["Hi", "Hello", "Hey"],
        "speakers": ["A", "B", "C"],
    }) is None


def test_everyday_conversations_uses_messages_and_falls_back_to_prompt_completion():
    messages = first_wave.convert_everyday_conversations_row({
        "messages": [
            {"role": "system", "content": "Ignore source persona."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hello! How are you?"},
        ],
    })
    assert messages == [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hello! How are you?"},
    ]

    fallback = first_wave.convert_everyday_conversations_row({
        "prompt": "Say hello.", "completion": "Hello!"
    })
    assert fallback == [
        {"role": "user", "content": "Say hello."},
        {"role": "assistant", "content": "Hello!"},
    ]


def test_english_daily_dialogues_preserves_multi_turn_order():
    messages = first_wave.convert_english_daily_dialogues_row({
        "turns": [
            {"speaker": "A", "text": "Want to get coffee?"},
            {"speaker": "B", "text": "Sure, what time?"},
            {"speaker": "A", "text": "Around noon."},
            {"speaker": "B", "text": "That works."},
        ],
    })
    assert [item["role"] for item in messages] == [
        "user", "assistant", "user", "assistant"
    ]


def test_synthetic_social_dialogues_parses_output_json():
    row = {
        "output": json.dumps({
            "inferred_scenario": "Two friends catch up.",
            "dialogue": [
                {"speaker": "Mina", "line": "How have you been?"},
                {"speaker": "Rae", "line": "Pretty good, thanks!"},
            ],
        }),
    }
    messages = first_wave.convert_synthetic_social_dialogues_row(row)
    assert messages == [
        {"role": "user", "content": "How have you been?"},
        {"role": "assistant", "content": "Pretty good, thanks!"},
    ]


def test_synthetic_conversations_accepts_human_and_gpt_roles():
    messages = first_wave.convert_synthetic_conversations_row({
        "conversations": [
            {"from": "human", "value": "I feel stuck."},
            {"from": "gpt", "value": "What feels most difficult right now?"},
            {"from": "human", "value": "Choosing a direction."},
            {"from": "gpt", "value": "Let's break the decision into smaller parts."},
        ],
    })
    assert len(messages) == 4
    assert messages[-1]["role"] == "assistant"


def test_multiwoz_dialogue_converts_user_and_system_turns():
    messages = first_wave.convert_multiwoz_dialogue({
        "dialogue_id": "test-1",
        "turns": [
            {"speaker": "USER", "utterance": "I need a hotel."},
            {"speaker": "SYSTEM", "utterance": "What area would you prefer?"},
        ],
    })
    assert messages == [
        {"role": "user", "content": "I need a hotel."},
        {"role": "assistant", "content": "What area would you prefer?"},
    ]


def test_multiwoz_rejects_unknown_speaker_and_checksum_catalog_is_complete():
    assert first_wave.convert_multiwoz_dialogue({
        "turns": [{"speaker": "TOOL", "utterance": "unexpected"}]
    }) is None
    assert len(first_wave.MULTIWOZ_TRAIN_SHARDS) == 17
    assert all(len(value) == 64 for value in first_wave.MULTIWOZ_TRAIN_SHARDS.values())


def test_record_preserves_license_and_attribution_provenance():
    record = first_wave.make_record(
        "example",
        [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}],
        repository="publisher/dataset",
        license_name="CC-BY-4.0",
        attribution="Original publisher",
        split="train",
        source_id="row-7",
        notes="Synthetic example.",
    )
    assert record["dataset_license"] == "CC-BY-4.0"
    assert record["dataset_attribution"] == "Original publisher"
    assert record["dataset_record_id"] == "row-7"
    assert record["dataset_repository"] == "publisher/dataset"
