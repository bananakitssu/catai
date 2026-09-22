from __future__ import annotations

import gzip
import json
import urllib.error
import urllib.request

from catai.oasst1 import PERSONALITY_SYSTEM_PROMPT, convert_dataset, download_dataset


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


def test_download_dataset_retries_http_429_and_honors_retry_after(tmp_path, monkeypatch):
    output = tmp_path / "oasst1_ready.trees.jsonl.gz"
    calls = 0
    sleeps: list[float] = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self, _size):
            return b"oasst1"

    def fake_urlopen(request, timeout):
        nonlocal calls
        assert request.full_url == "https://example.com/oasst1.gz"
        assert timeout == 120
        calls += 1
        if calls < 3:
            raise urllib.error.HTTPError(
                request.full_url,
                429,
                "Too Many Requests",
                {"Retry-After": "2"},
                None,
            )
        return Response()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr("catai.oasst1.time.sleep", sleeps.append)

    download_dataset("https://example.com/oasst1.gz", output)

    assert calls == 3
    assert sleeps == [2.0, 2.0]
    assert output.read_bytes() == b"oasst1"
    assert not output.with_name(output.name + ".part").exists()


def test_download_dataset_does_not_retry_non_retryable_http_error(tmp_path, monkeypatch):
    output = tmp_path / "oasst1_ready.trees.jsonl.gz"
    calls = 0

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        raise urllib.error.HTTPError(
            request.full_url,
            404,
            "Not Found",
            {},
            None,
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    try:
        download_dataset("https://example.com/oasst1.gz", output)
    except urllib.error.HTTPError as error:
        assert error.code == 404
    else:
        raise AssertionError("expected HTTP 404")

    assert calls == 1
    assert not output.exists()
    assert not output.with_name(output.name + ".part").exists()
