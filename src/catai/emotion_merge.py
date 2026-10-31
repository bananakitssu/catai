"""Merge source-specific emotion datasets into one auditable CatAI SFT file."""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
from typing import Any
from .chat_dataset import validate_messages, validate_state

def _read_source(path: Path, source: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"emotion dataset file does not exist: {path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_number}: record must be a JSON object")
        if record.get("dataset_source") != source:
            raise ValueError(
                f"{path}:{line_number}: expected dataset_source={source!r}, got {record.get('dataset_source')!r}"
            )
        if record.get("dataset_license") is None:
            raise ValueError(f"{path}:{line_number}: missing dataset license metadata")
        record["state"] = validate_state(record.get("state"))
        record["messages"] = [
            {"role": message["role"], "content": message["content"].strip()}
            for message in validate_messages(record.get("messages"))
        ]
        records.append(record)
    if not records:
        raise ValueError(f"{source} contained zero usable examples")
    return records

def merge_emotion_datasets(
    go_emotions_path: str | Path,
    daily_dialog_path: str | Path,
    empathetic_dialogues_path: str | Path,
    output_path: str | Path,
    *,
    seed: int = 42,
    max_examples: int | None = None,
) -> dict[str, int]:
    """Validate, deduplicate, and deterministically merge all three emotion sources."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if max_examples is not None and max_examples < 1:
        raise ValueError("max_examples must be positive when provided")
    sources = [
        ("go_emotions", Path(go_emotions_path)),
        ("daily_dialog", Path(daily_dialog_path)),
        ("empathetic_dialogues", Path(empathetic_dialogues_path)),
    ]
    seen: set[str] = set()
    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    duplicates = 0
    for source, path in sources:
        source_records = _read_source(path, source)
        unique: list[dict[str, Any]] = []
        for record in source_records:
            conversation = [
                {"role": item["role"], "content": item["content"]}
                for item in record["messages"] if item["role"] != "system"
            ]
            fingerprint = json.dumps(
                conversation, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            if fingerprint in seen:
                duplicates += 1
                continue
            seen.add(fingerprint)
            unique.append(record)
        counts[source] = len(unique)
        selected.extend(unique)
    if not selected:
        raise ValueError("all three emotion datasets contained zero unique examples")
    rng = random.Random(seed)
    rng.shuffle(selected)
    if max_examples is not None:
        selected = selected[:max_examples]
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
    return {**counts, "duplicates_removed": duplicates, "total": len(selected)}

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Merge GoEmotions, DailyDialog, and EmpatheticDialogues for CatAI.")
    parser.add_argument("--go-emotions", type=Path, default=Path("data/go_emotions_emotion.jsonl"))
    parser.add_argument("--daily-dialog", type=Path, default=Path("data/daily_dialog_emotion.jsonl"))
    parser.add_argument("--empathetic-dialogues", type=Path, default=Path("data/empathetic_dialogues_emotion.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/catai_emotion.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-examples", type=int, default=None)
    args = parser.parse_args(argv)
    stats = merge_emotion_datasets(
        args.go_emotions, args.daily_dialog, args.empathetic_dialogues, args.output,
        seed=args.seed, max_examples=args.max_examples,
    )
    print(
        f"Merged {stats['total']:,} emotional examples to {args.output}: "
        f"GoEmotions {stats['go_emotions']:,}, DailyDialog {stats['daily_dialog']:,}, "
        f"EmpatheticDialogues {stats['empathetic_dialogues']:,}; "
        f"{stats['duplicates_removed']:,} duplicates removed."
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
