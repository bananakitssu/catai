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


def test_two_speaker_dialogue_drops_incomplete_final_user_turn():
    messages = first_wave._normalize_two_speaker_turns(
        [
            {"speaker": "A", "text": "Hello."},
            {"speaker": "B", "text": "Hi!"},
            {"speaker": "A", "text": "One more question."},
        ],
        speaker_key="speaker",
        text_key="text",
    )
    assert messages == [
        {"role": "user", "content": "Hello."},
        {"role": "assistant", "content": "Hi!"},
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
    assert first_wave.MULTIWOZ_TRAIN_SHARDS == {
        "dialogues_001.json": "2d1a5d92bb5ea1a6017f416f76b43e11bdad945b1f93acf870f2f671a98c3032",
        "dialogues_002.json": "2f3ea771d4e01cb2780357738cff7f7496b87d34c221cc240df74501312438d3",
        "dialogues_003.json": "74f73311936c7b4928b073698e5ea61dedad009dc45672963735684e1279521a",
        "dialogues_004.json": "2f9f18b6cb0abbc58c86ca9291aa19baf80395521f9556cf02189c698c48bc52",
        "dialogues_005.json": "eedbe5c649072a3c163e8da109f3c3fd0502425c63db584a11a9cadaa09c91db",
        "dialogues_006.json": "2a51cfed1224110e4cdce80b051d3f7ebecd223d0b3d1c4ade0dd8fc484675fe",
        "dialogues_007.json": "e062becf19b670a45b9333390b6a72f93fc2ed436371ba342499e6532912b1f3",
        "dialogues_008.json": "9cd73cfae130e4d8c76854b6c0f393d68fb084f665a1c7527e04e3b26f877d1b",
        "dialogues_009.json": "fff3f55ee9bd8ecc2492cdb8101e3d5908705e4edd65b3e8cbd7a825be18e47e",
        "dialogues_010.json": "628db438d4f6ab62fb849d3e9df264845b921961052c19b336c8853e2bf4e8d8",
        "dialogues_011.json": "82e2d2900a037b866a01d05974734dd419e89b329fe29ef93b35eea96d27feb8",
        "dialogues_012.json": "5b18461e0f3ff0698818e02a2e105f4f25727d88d02cc4706de19d138e9ceba0",
        "dialogues_013.json": "0f90ba730400be770f753fc46e868189709e6ce06e97d780843ed09db16762b4",
        "dialogues_014.json": "7bd668216f43f6713232c07dcaa1d84e490eeee15caa1b3d843c3075d4b84a10",
        "dialogues_015.json": "5f551d134860595e7834eedab3ebe16e277953cc81b29f46104ef3070cddd4ac",
        "dialogues_016.json": "22eeed01c15a4fba6ac395a6708400672c5c07dbfbd6448bcb451f8697324881",
        "dialogues_017.json": "a1dba37c0fd1dba519500b57be0384219dd4a7bacaec94e124d431932724bd8d",
    }


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



def test_prepare_builds_all_six_sources_and_appends_to_existing_sft(tmp_path, monkeypatch):
    source_rows = {
        "allenai/soda": [{
            "id": "soda-1",
            "dialogue": ["I need directions.", "Where are you going?"],
            "speakers": ["A", "B"],
        }],
        "HuggingFaceTB/everyday-conversations-llama3.1-2k": [{
            "id": "everyday-1",
            "messages": [
                {"role": "user", "content": "Good morning."},
                {"role": "assistant", "content": "Good morning!"},
            ],
        }],
        "3nesdeniz/english-daily-dialogues-10k": [{
            "id": 11,
            "turns": [
                {"speaker": "A", "text": "Are you free today?"},
                {"speaker": "B", "text": "Yes, after lunch."},
            ],
        }],
        "AppliedLucent/synthetic_conversations": [{
            "id": "synthetic-1",
            "conversations": [
                {"from": "human", "value": "I want to learn Python."},
                {"from": "gpt", "value": "Let's start with variables."},
            ],
        }],
    }

    def fake_sample(repository, config, split, limit, seed):
        return source_rows[repository]

    monkeypatch.setattr(first_wave, "sample_viewer_rows", fake_sample)
    monkeypatch.setattr(
        first_wave,
        "_download_multiwoz_examples",
        lambda limit, seed: [
            first_wave.make_record(
                "multiwoz",
                [
                    {"role": "user", "content": "Find a restaurant."},
                    {"role": "assistant", "content": "What area do you prefer?"},
                ],
                repository=first_wave.MULTIWOZ_SPEC["repository"],
                license_name=first_wave.MULTIWOZ_SPEC["license"],
                attribution=first_wave.MULTIWOZ_SPEC["attribution"],
                split="train",
            )
        ],
    )
    monkeypatch.setattr(
        first_wave,
        "_download_social_dialogue_rows",
        lambda limit, seed: [{
            "id": "social-1",
            "output": json.dumps({
                "dialogue": [
                    {"speaker": "Rin", "line": "How was your weekend?"},
                    {"speaker": "Kai", "line": "Really relaxing, thanks."},
                ],
            }),
        }],
    )

    existing_path = tmp_path / "existing_sft.jsonl"
    existing_path.write_text(
        json.dumps({
            "state": first_wave.default_state(),
            "messages": [
                {"role": "user", "content": "Existing prompt."},
                {"role": "assistant", "content": "Existing answer."},
            ],
            "dataset_source": "existing",
        }) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "first_wave.jsonl"
    manifest_path = tmp_path / "manifest.json"

    manifest = first_wave.prepare(
        output=output,
        manifest_output=manifest_path,
        max_examples_per_source=2,
        seed=42,
        append_to=existing_path,
    )

    first_wave_records = [
        json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
    ]
    merged_records = [
        json.loads(line) for line in existing_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(first_wave_records) == 6
    assert {item["dataset_source"] for item in first_wave_records} == {
        "soda",
        "everyday_conversations",
        "english_daily_dialogues",
        "synthetic_conversations",
        "multiwoz",
        "synthetic_social_dialogues",
    }
    assert len(merged_records) == 7
    assert manifest["total_sources"] == 6
    assert manifest["outputs"]["merged_sft_examples"] == 7
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["total_sources"] == 6
