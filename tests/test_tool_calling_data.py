from __future__ import annotations

import json
import re
from pathlib import Path

from catai.chat_dataset import ASSISTANT_OUTPUT_ROLES, load_chat_dataset


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "instruction_chat.jsonl"
TOOL_ENVELOPE = re.compile(r"^<tool_call>\s*(\{.*\})\s*</tool_call>$", re.DOTALL)
MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")


def _raw_examples():
    rows = DATASET.read_text(encoding="utf-8").splitlines()
    return [json.loads(row) for row in rows if row.strip()]


def test_tool_examples_use_the_runtime_envelope_and_role_trace():
    examples = _raw_examples()
    found = 0

    for example in examples:
        messages = example["messages"]
        call_names = {
            tool["name"] for tool in example.get("tools", [])
        }
        for index, call_message in enumerate(messages):
            if call_message["role"] not in {"tool_call:assistant", "mcp_tool_call:assistant"}:
                continue

            payload = json.loads(call_message["content"])
            assert set(payload) == {"tool", "args"}
            assert isinstance(payload["tool"], str) and payload["tool"]
            assert isinstance(payload["args"], dict)
            assert payload["tool"] in call_names

            matching = [
                message for message in messages[:index]
                if message["role"] == "assistant"
                and TOOL_ENVELOPE.fullmatch(message["content"])
            ]
            assert matching, "a tool trace should have a matching generation envelope"
            envelope = json.loads(TOOL_ENVELOPE.fullmatch(matching[-1]["content"]).group(1))
            assert envelope == payload
            found += 1

    assert found >= 55
    assert "tool_call:assistant" in ASSISTANT_OUTPUT_ROLES
    assert "mcp_tool_call:assistant" in ASSISTANT_OUTPUT_ROLES


def test_cited_answers_link_to_urls_returned_by_web_tools():
    examples = _raw_examples()
    citation_examples = 0

    for example in examples:
        messages = example["messages"]
        result_text = "\n".join(
            message["content"] for message in messages
            if message["role"] in {"tool_call:result->assistant", "mcp_tool_call:result->assistant"}
        )
        if "http" not in result_text.lower():
            continue
        assistant_answers = [
            message["content"] for message in messages
            if message["role"] == "assistant"
            and not TOOL_ENVELOPE.fullmatch(message["content"])
        ]
        for answer in assistant_answers:
            for _, url in MARKDOWN_LINK.findall(answer):
                assert url in result_text, f"citation URL must come from tool output: {url}"
                citation_examples += 1

    assert citation_examples >= 20


def test_curated_dataset_contains_both_direct_answers_and_tool_use():
    examples = load_chat_dataset(DATASET)
    assert any(
        any(message["role"] == "assistant" for message in example)
        and not any(message["role"] in {"tool_call:assistant", "mcp_tool_call:assistant"} for message in example)
        for example in examples
    )
    assert any(any(message["role"] == "tool_call:assistant" for message in example) for example in examples)
    assert any(any(message["role"] == "mcp_tool_call:assistant" for message in example) for example in examples)

def test_tool_metadata_is_outside_messages_and_rendered_as_system_context():
    raw_examples = _raw_examples()
    examples = load_chat_dataset(DATASET)

    assert not any(
        message["role"] == "file"
        for example in raw_examples
        for message in example["messages"]
    )
    tool_examples = [
        example for example in raw_examples if "tools" in example
    ]
    assert len(tool_examples) >= 60
    assert any(example["tools"] for example in tool_examples)
    assert any(example["tools"] == [] for example in tool_examples)

    for raw_example, rendered_messages in zip(raw_examples, examples):
        if "tools" not in raw_example:
            continue
        system_prompt = rendered_messages[0]["content"]
        if raw_example["tools"]:
            assert "Tools available for this message" in system_prompt
            for tool in raw_example["tools"]:
                assert tool["name"] in system_prompt
                assert tool["description"] in system_prompt
                for argument, schema in tool["parameters"]["properties"].items():
                    argument_type = schema.get("type", "any")
                    assert f'"{argument}":"{argument_type}"' in system_prompt
        else:
            assert "No tools are enabled" in system_prompt


def test_each_training_tool_call_is_listed_in_that_examples_tool_metadata():
    for example in _raw_examples():
        enabled = {tool["name"] for tool in example.get("tools", [])}
        for message in example["messages"]:
            if message["role"] not in {"tool_call:assistant", "mcp_tool_call:assistant"}:
                continue
            payload = json.loads(message["content"])
            assert payload["tool"] in enabled
