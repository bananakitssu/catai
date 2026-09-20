import json

import pytest

from catai.chat_dataset import load_chat_dataset, validate_messages


def test_validate_messages_accepts_chat_roles():
    messages = validate_messages(
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi!"},
        ]
    )

    assert messages[-1] == {"role": "assistant", "content": "Hi!"}


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "user", "content": "Hello"}],
        [{"role": "developer", "content": "Hello"}],
        [{"role": "assistant", "content": ""}],
    ],
)
def test_validate_messages_rejects_invalid_examples(messages):
    with pytest.raises(ValueError):
        validate_messages(messages)


def test_load_chat_dataset_validates_jsonl(tmp_path):
    dataset = tmp_path / "chat.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "2 + 2?"},
                    {"role": "assistant", "content": "4"},
                ]
            }
        )
        + "\n"
    )

    examples = load_chat_dataset(dataset)

    assert examples == [
        [
            {"role": "user", "content": "2 + 2?"},
            {"role": "assistant", "content": "4"},
        ]
    ]


def test_load_chat_dataset_rejects_invalid_json(tmp_path):
    dataset = tmp_path / "chat.jsonl"
    dataset.write_text("{not-json}\n")

    with pytest.raises(ValueError, match="invalid JSON on line 1"):
        load_chat_dataset(dataset)
