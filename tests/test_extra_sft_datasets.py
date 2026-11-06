from data.prepare_extra_sft_datasets import (
    BLENDED_SKILL_TALK_SHA256,
    convert_blended_skill_talk_row,
    convert_smol_smoltalk_row,
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
