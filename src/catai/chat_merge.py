"""Deterministic, deduplicated OASST1 + OASST2 SFT dataset preparation."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def _read_records(path: Path, source: str, seen: set[str]) -> tuple[list[dict[str, Any]], int]:
    """Read validated CatAI chat examples and remove duplicate conversations."""
    if not path.is_file():
        raise FileNotFoundError(f"dataset file does not exist: {path}")

    records: list[dict[str, Any]] = []
    duplicates = 0
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc

            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: record must be a JSON object")
            if not isinstance(record.get("state"), dict):
                raise ValueError(f"{path}:{line_number}: missing state object")
            messages = record.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError(f"{path}:{line_number}: messages must contain a complete exchange")

            normalized: list[dict[str, str]] = []
            for message in messages:
                if not isinstance(message, dict):
                    raise ValueError(f"{path}:{line_number}: each message must be an object")
                role = message.get("role")
                content = message.get("content")
                if not isinstance(role, str) or not role:
                    raise ValueError(f"{path}:{line_number}: message role must be non-empty")
                if not isinstance(content, str) or not content.strip():
                    raise ValueError(f"{path}:{line_number}: message content must be non-empty")
                normalized.append({"role": role, "content": content.strip()})

            roles = [message["role"] for message in normalized]
            if roles[0] != "user" or roles[-1] != "assistant":
                raise ValueError(f"{path}:{line_number}: conversation must start with user and end with assistant")
            if any(left == right for left, right in zip(roles, roles[1:])):
                raise ValueError(f"{path}:{line_number}: conversation roles must alternate")

            # Deduplicate on message contents, not state annotations or source metadata.
            fingerprint = json.dumps(
                normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            if fingerprint in seen:
                duplicates += 1
                continue
            seen.add(fingerprint)

            record["messages"] = normalized
            record["dataset_source"] = source
            records.append(record)

    return records, duplicates


def _balanced_selection(
    oasst1: list[dict[str, Any]],
    oasst2: list[dict[str, Any]],
    max_examples: int | None,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], int, int]:
    """Sample both corpora evenly, filling unused quota from the other source."""
    rng.shuffle(oasst1)
    rng.shuffle(oasst2)

    if max_examples is None or max_examples >= len(oasst1) + len(oasst2):
        first_count, second_count = len(oasst1), len(oasst2)
    else:
        limit = max_examples
        first_count = min(len(oasst1), (limit + 1) // 2)
        second_count = min(len(oasst2), limit // 2)
        remaining = limit - first_count - second_count
        while remaining > 0:
            first_capacity = len(oasst1) - first_count
            second_capacity = len(oasst2) - second_count
            if first_capacity <= 0 and second_capacity <= 0:
                break
            if first_capacity >= second_capacity and first_capacity > 0:
                first_count += 1
            else:
                second_count += 1
            remaining -= 1

    selected = oasst1[:first_count] + oasst2[:second_count]
    rng.shuffle(selected)
    return selected, first_count, second_count


def merge_chat_datasets(
    oasst1_path: str | Path,
    oasst2_path: str | Path,
    output_path: str | Path,
    *,
    max_examples: int | None = 4500,
    seed: int = 42,
) -> dict[str, int]:
    """Merge, deduplicate, balance, and deterministically shuffle both datasets."""
    if max_examples is not None and max_examples < 1:
        raise ValueError("max_examples must be positive when provided")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    seen: set[str] = set()
    first, first_duplicates = _read_records(Path(oasst1_path), "oasst1", seen)
    second, second_duplicates = _read_records(Path(oasst2_path), "oasst2", seen)
    if not first and not second:
        raise ValueError("both OpenAssistant datasets contained zero usable conversations")
    rng = random.Random(seed)
    selected, first_count, second_count = _balanced_selection(
        first, second, max_examples, rng
    )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            for record in selected:
                stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
                stream.write("\n")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)

    return {
        "oasst1_available": len(first),
        "oasst2_available": len(second),
        "duplicates_removed": first_duplicates + second_duplicates,
        "oasst1_selected": first_count,
        "oasst2_selected": second_count,
        "total": len(selected),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Merge English OASST1 and OASST2 conversations for CatAI SFT."
    )
    parser.add_argument("--oasst1", type=Path, default=Path("data/oasst1_chat.jsonl"))
    parser.add_argument("--oasst2", type=Path, default=Path("data/oasst2_chat.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/oasst_chat.jsonl"))
    parser.add_argument(
        "--max-examples",
        type=int,
        default=4500,
        help="Maximum combined examples, split evenly between sources where possible.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    stats = merge_chat_datasets(
        args.oasst1,
        args.oasst2,
        args.output,
        max_examples=args.max_examples,
        seed=args.seed,
    )
    print(
        f"Merged {stats['total']:,} examples to {args.output}: "
        f"OASST1 {stats['oasst1_selected']:,}, "
        f"OASST2 {stats['oasst2_selected']:,}; "
        f"{stats['duplicates_removed']:,} duplicates removed "
        f"({stats['oasst1_available']:,} and {stats['oasst2_available']:,} unique candidates)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
