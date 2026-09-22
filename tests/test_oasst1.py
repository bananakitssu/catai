from __future__ import annotations

import gzip
import json

from catai.oasst1 import PERSONALITY_SYSTEM_PROMPT, convert_dataset


def test_convert_oasst1_tree_to_catai_jsonl(tmp_path):
    source = tmp_path / "trees.jsonl.gz"
    output = tmp_path / "chat.jsonl"

    tree = {
        "message_tree_id": "tree-1",
        "tree_state": "ready_for_export",
        "prompt": {
            "message_id": "root",
            "text": "Hello!",
            "role": "prompter",
            "lang": "en",
            "replies": [
                {
                    "message_id": "a1",
                    "text": "Hi there!",
                    "role": "assistant",
                    "lang": "en",
                    "replies": [
                        {
                            "message_id": "u2",
                            "text": "How are you?",
                            "role": "prompter",
                            "lang": "en",
                            "replies": [
                                {
                                    "message_id": "a2",
                                    "text": "I'm doing well!",
                                    "role": "assistant",
                                    "lang": "en",
                                    "replies": [],
                                }
                            ],
                        }
                    ],
                }
            ],
        },
    }

    with gzip.open(source, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(tree) + "\n")

    trees, examples = convert_dataset(
        source, output, language="en", max_trees=None, max_examples=None
    )

    assert (trees, examples) == (1, 1)
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["messages"][0] == {
        "role": "system",
        "content": PERSONALITY_SYSTEM_PROMPT,
    }
    assert record["messages"][1:] == [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "I'm doing well!"},
    ]


def test_convert_oasst1_filters_non_english_paths(tmp_path):
    source = tmp_path / "trees.jsonl.gz"
    output = tmp_path / "chat.jsonl"

    tree = {
        "prompt": {
            "text": "Hola",
            "role": "prompter",
            "lang": "es",
            "replies": [
                {
                    "text": "Hola!",
                    "role": "assistant",
                    "lang": "es",
                    "replies": [],
                }
            ],
        }
    }

    with gzip.open(source, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(tree) + "\n")

    trees, examples = convert_dataset(
        source, output, language="en", max_trees=None, max_examples=None
    )

    assert (trees, examples) == (1, 0)
    assert output.read_text(encoding="utf-8") == ""
