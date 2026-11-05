from data.prepare_extra_sft_datasets import (
    convert_blended_skill_talk_row,
    convert_smol_smoltalk_row,
)


def test_smol_smoltalk_provenance_and_filters():
    base = {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hello!"},
        ],
        "conversation_tokens": 120,
        "quality": "good",
        "source": "smol-magpie-ultra-short",
    }
    item = convert_smol_smoltalk_row(base)
    assert item is not None
    assert item["dataset_license"] == "Apache-2.0"
    assert item["upstream_source"] == "smol-magpie-ultra-short"
    assert convert_smol_smoltalk_row({**base, "conversation_tokens": 1500}) is None
    assert convert_smol_smoltalk_row({**base, "quality": "bad"}) is None


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
