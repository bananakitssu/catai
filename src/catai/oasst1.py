"""OASST1 conversion helpers for CatAI supervised fine-tuning."""

from __future__ import annotations

import argparse
import email.utils
import gzip
import json
import socket
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

DEFAULT_URL = (
    "https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/main/"
    "2023-04-12_oasst_ready.trees.jsonl.gz"
)

PERSONALITY_SYSTEM_PROMPT = (
    "You are CatAI, a friendly cat-themed conversational AI. "
    "Be helpful, clear, concise, and playful when appropriate. "
    "Use light cat-like expressions such as ':3' occasionally, but do not force "
    "them into every response. Be honest when uncertain and never invent facts."
)

DOWNLOAD_MAX_ATTEMPTS = 6
DOWNLOAD_INITIAL_BACKOFF = 15.0
DOWNLOAD_MAX_BACKOFF = 300.0
RETRYABLE_HTTP_STATUS = frozenset({429, 500, 502, 503, 504})


def _retry_after_seconds(error: urllib.error.HTTPError) -> float | None:
    """Return a server-requested retry delay when Retry-After is present."""
    value = error.headers.get("Retry-After")
    if value is None:
        return None

    try:
        return max(0.0, float(value))
    except ValueError:
        pass

    try:
        retry_at = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        return None

    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)

    return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())


def _download_retry_delay(
    attempt: int, error: urllib.error.HTTPError | None = None
) -> float:
    """Choose a bounded retry delay, honoring Retry-After when available."""
    if error is not None:
        retry_after = _retry_after_seconds(error)
        if retry_after is not None:
            return min(retry_after, DOWNLOAD_MAX_BACKOFF)

    return min(
        DOWNLOAD_INITIAL_BACKOFF * (2 ** (attempt - 1)),
        DOWNLOAD_MAX_BACKOFF,
    )


def _is_retryable_http_error(error: urllib.error.HTTPError) -> bool:
    return error.code in RETRYABLE_HTTP_STATUS


def download_dataset(url: str, output: Path) -> None:
    """Download OASST1's ready-for-export conversation trees with retries."""
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".part")

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "CatAI-oasst1-converter/1.1"},
    )

    try:
        for attempt in range(1, DOWNLOAD_MAX_ATTEMPTS + 1):
            try:
                with urllib.request.urlopen(request, timeout=120) as response, partial.open(
                    "wb"
                ) as destination:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        destination.write(chunk)

                partial.replace(output)
                return

            except urllib.error.HTTPError as error:
                if not _is_retryable_http_error(error):
                    raise
                if attempt == DOWNLOAD_MAX_ATTEMPTS:
                    raise

                delay = _download_retry_delay(attempt, error)
                print(
                    f"OASST1 download hit HTTP {error.code}; "
                    f"retrying in {delay:.0f}s "
                    f"(attempt {attempt}/{DOWNLOAD_MAX_ATTEMPTS})..."
                )
                time.sleep(delay)

            except (urllib.error.URLError, TimeoutError, socket.timeout):
                if attempt == DOWNLOAD_MAX_ATTEMPTS:
                    raise

                delay = _download_retry_delay(attempt)
                print(
                    "OASST1 download failed due to a network error; "
                    f"retrying in {delay:.0f}s "
                    f"(attempt {attempt}/{DOWNLOAD_MAX_ATTEMPTS})..."
                )
                time.sleep(delay)
    finally:
        try:
            partial.unlink()
        except FileNotFoundError:
            pass


def _role(role: object) -> str | None:
    if role == "prompter":
        return "user"
    if role == "assistant":
        return "assistant"
    return None


def _node_is_usable(node: object, language: str | None) -> bool:
    if not isinstance(node, dict):
        return False

    text = node.get("text")
    if not isinstance(text, str) or not text.strip():
        return False

    if _role(node.get("role")) is None:
        return False

    if language is not None and node.get("lang") != language:
        return False

    if node.get("deleted") is True or node.get("review_result") is False:
        return False

    return True


def _leaf_paths(
    node: dict[str, object],
    path: tuple[dict[str, object], ...] = (),
    *,
    language: str | None,
) -> Iterator[tuple[dict[str, object], ...]]:
    """Yield every usable root-to-leaf path in one OASST1 tree."""
    if not _node_is_usable(node, language):
        return

    current = path + (node,)
    replies = node.get("replies", [])
    if not isinstance(replies, list):
        replies = []

    usable_replies = [
        reply
        for reply in replies
        if isinstance(reply, dict) and _node_is_usable(reply, language)
    ]

    if not usable_replies:
        yield current
        return

    for reply in usable_replies:
        yield from _leaf_paths(reply, current, language=language)


def convert_path(path: tuple[dict[str, object], ...]) -> dict[str, list[dict[str, str]]] | None:
    """Convert one OASST1 root-to-leaf path to CatAI's messages format."""
    messages: list[dict[str, str]] = [
        {"role": "system", "content": PERSONALITY_SYSTEM_PROMPT}
    ]

    for node in path:
        role = _role(node.get("role"))
        text = node.get("text")
        if role is None or not isinstance(text, str) or not text.strip():
            return None
        messages.append({"role": role, "content": text.strip()})

    if not any(message["role"] == "assistant" for message in messages):
        return None

    # OASST1 alternates prompter/assistant; reject malformed paths rather than
    # silently teaching CatAI an unexpected role order.
    roles = [message["role"] for message in messages[1:]]
    if not roles or roles[0] != "user":
        return None
    if any(left == right for left, right in zip(roles, roles[1:])):
        return None

    # SFT examples must represent a completed exchange. A path ending in a
    # user message has no desired response for that final prompt, and long
    # examples can otherwise have all assistant targets truncated out.
    if roles[-1] != "assistant":
        return None

    return {"messages": messages}


def convert_dataset(
    source: Path,
    output: Path,
    *,
    language: str | None = "en",
    max_trees: int | None = None,
    max_examples: int | None = None,
) -> tuple[int, int]:
    """Convert an OASST1 tree archive to CatAI JSONL.

    Returns (trees_seen, examples_written).
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    trees_seen = 0
    examples_written = 0

    with gzip.open(source, "rt", encoding="utf-8") as source_file, output.open(
        "w", encoding="utf-8"
    ) as destination:
        for line in source_file:
            if not line.strip():
                continue
            if max_trees is not None and trees_seen >= max_trees:
                break

            record = json.loads(line)
            trees_seen += 1
            root = record.get("prompt") if isinstance(record, dict) else None
            if not isinstance(root, dict):
                continue

            for path in _leaf_paths(root, language=language):
                converted = convert_path(path)
                if converted is None:
                    continue

                destination.write(
                    json.dumps(converted, ensure_ascii=False, separators=(",", ":"))
                )
                destination.write("\n")
                examples_written += 1

                if max_examples is not None and examples_written >= max_examples:
                    return trees_seen, examples_written

    return trees_seen, examples_written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and convert OASST1 into CatAI chat JSONL."
    )
    parser.add_argument("--output", type=Path, default=Path("data/oasst1_chat.jsonl"))
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/oasst1_ready.trees.jsonl.gz"),
        help="Local OASST1 archive; downloaded automatically when missing.",
    )
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument(
        "--language",
        default="en",
        help="Language tag to keep; use 'all' for every language.",
    )
    parser.add_argument("--max-trees", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument(
        "--keep-source",
        action="store_true",
        help="Keep the downloaded .jsonl.gz archive after conversion.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.max_trees is not None and args.max_trees < 1:
        raise ValueError("--max-trees must be positive")
    if args.max_examples is not None and args.max_examples < 1:
        raise ValueError("--max-examples must be positive")

    language = None if args.language == "all" else args.language

    if not args.source.exists():
        print(f"Downloading OASST1 ready trees to {args.source}...")
        download_dataset(args.url, args.source)

    trees_seen, examples_written = convert_dataset(
        args.source,
        args.output,
        language=language,
        max_trees=args.max_trees,
        max_examples=args.max_examples,
    )

    if not args.keep_source:
        try:
            args.source.unlink()
        except FileNotFoundError:
            pass

    print(
        f"Wrote {examples_written:,} CatAI chat examples from "
        f"{trees_seen:,} OASST1 trees to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
