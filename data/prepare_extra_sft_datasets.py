#!/usr/bin/env python3
"""Prepare bounded Smol-SmolTalk and Blended Skill Talk SFT samples."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import tarfile
import urllib.request
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from catai.chat_dataset import default_state, validate_messages

SMOL_SMOLTALK_DATASET = "HuggingFaceTB/smol-smoltalk"
BLENDED_SKILL_TALK_URL = "https://parl.ai/downloads/blended_skill_talk/blended_skill_talk.tar.gz"
BLENDED_SKILL_TALK_SHA256 = "5fbed0068ee89e2d43b93c3ecb341e784617033efa5e8e911a2194deda6134a6"


def normalize_messages(raw_messages: object) -> list[dict[str, str]] | None:
    if isinstance(raw_messages, str):
        try:
            raw_messages = json.loads(raw_messages)
        except json.JSONDecodeError:
            return None
    if not isinstance(raw_messages, list):
        return None
    aliases = {"human": "user", "gpt": "assistant", "bot": "assistant"}
    messages = []
    for item in raw_messages:
        if not isinstance(item, dict):
            continue
        role, content = item.get("role"), item.get("content")
        if not isinstance(role, str) or not isinstance(content, str) or not content.strip():
            continue
        role = aliases.get(role.lower(), role.lower())
        if role in {"system", "user", "assistant"}:
            messages.append({"role": role, "content": content.strip()})
    while messages and messages[-1]["role"] != "assistant":
        messages.pop()
    if not any(m["role"] == "user" for m in messages):
        return None
    if not any(m["role"] == "assistant" for m in messages):
        return None
    try:
        return validate_messages(messages)
    except ValueError:
        return None


def convert_smol_smoltalk_row(row: dict[str, Any]) -> dict[str, Any] | None:
    messages = normalize_messages(row.get("messages"))
    if messages is None:
        return None
    try:
        token_count = int(row.get("conversation_tokens", 0))
    except (TypeError, ValueError):
        token_count = 0
    if token_count and token_count > 900:
        return None
    quality = str(row.get("quality", "")).lower()
    if quality not in {"good", "average", ""}:
        return None
    return {
        "state": default_state(),
        "messages": messages,
        "dataset_source": "smol-smoltalk",
        "dataset_license": "Apache-2.0",
        "dataset_repository": SMOL_SMOLTALK_DATASET,
        "upstream_source": str(row.get("source", "unknown")),
        "quality": quality or None,
    }


def _bst_turn(turn: object) -> tuple[str, str] | None:
    """ParlAI stores each turn as [speaker_id, utterance]."""
    if isinstance(turn, (list, tuple)) and len(turn) >= 2:
        speaker, content = turn[0], turn[1]
    elif isinstance(turn, dict):
        speaker = turn.get("id", turn.get("speaker"))
        content = turn.get("text", turn.get("content"))
    else:
        return None
    role = {0: "user", 1: "assistant", "0": "user", "1": "assistant"}.get(speaker)
    if role is None or not isinstance(content, str) or not content.strip():
        return None
    return role, content.strip()


def convert_blended_skill_talk_row(row: dict[str, Any]) -> dict[str, Any] | None:
    free = row.get("free_turker_utterance")
    guided = row.get("guided_turker_utterance")
    if not isinstance(free, str) or not free.strip() or not isinstance(guided, str) or not guided.strip():
        return None
    context_dataset = row.get("context_dataset")
    system_parts = []
    personas = row.get("personas")
    if isinstance(personas, list) and len(personas) > 1 and isinstance(personas[1], list):
        persona = [x.strip() for x in personas[1] if isinstance(x, str) and x.strip()]
        if persona:
            system_parts.append("Your persona: " + " ".join(persona))
    if isinstance(context_dataset, str) and context_dataset.strip():
        system_parts.append("Context source: " + context_dataset.strip())
    additional = row.get("additional_context")
    if context_dataset == "wizard_of_wikipedia" and isinstance(additional, str) and additional.strip():
        system_parts.append("Topic: " + additional.strip())

    messages = []
    if system_parts:
        messages.append({"role": "system", "content": "\n".join(system_parts)})
    messages.extend([
        {"role": "user", "content": free.strip()},
        {"role": "assistant", "content": guided.strip()},
    ])
    dialog = row.get("dialog")
    if not isinstance(dialog, list) or len(dialog) % 2:
        return None
    for index in range(0, len(dialog), 2):
        free_turn, guided_turn = _bst_turn(dialog[index]), _bst_turn(dialog[index + 1])
        if (
            free_turn is None or guided_turn is None
            or free_turn[0] != "user" or guided_turn[0] != "assistant"
        ):
            continue
        messages.extend([
            {"role": "user", "content": free_turn[1]},
            {"role": "assistant", "content": guided_turn[1]},
        ])
    try:
        messages = validate_messages(messages)
    except ValueError:
        return None
    return {
        "state": default_state(),
        "messages": messages,
        "dataset_source": "blended_skill_talk",
        "dataset_license": "CC-BY-4.0",
        "dataset_repository": "facebookresearch/ParlAI:blended_skill_talk",
        "context_dataset": str(context_dataset or "unknown"),
    }


def _write_jsonl(records: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".part")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_smol_smoltalk(output: Path, limit: int, seed: int) -> int:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Smol-SmolTalk requires datasets>=3.3") from exc
    dataset = load_dataset(SMOL_SMOLTALK_DATASET, split="train", streaming=True).shuffle(
        seed=seed, buffer_size=10_000
    )
    records = []
    for row in dataset:
        item = convert_smol_smoltalk_row(row)
        if item is not None:
            records.append(item)
            if len(records) >= limit:
                break
    if not records:
        raise RuntimeError("Smol-SmolTalk produced no usable examples")
    _write_jsonl(records, output)
    print(f"Prepared {len(records):,} Smol-SmolTalk examples at {output}")
    return len(records)


def prepare_blended_skill_talk(output: Path, limit: int, seed: int) -> int:
    request = urllib.request.Request(
        BLENDED_SKILL_TALK_URL,
        headers={"User-Agent": "CatAI-SFT-dataset-preparation/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    if hashlib.sha256(payload).hexdigest() != BLENDED_SKILL_TALK_SHA256:
        raise RuntimeError("Blended Skill Talk archive checksum mismatch")
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        matches = [
            member for member in archive.getmembers()
            if member.isfile() and member.name.rstrip("/").endswith("train.json")
        ]
        if not matches:
            raise ValueError("Blended Skill Talk archive does not contain train.json")
        source = archive.extractfile(matches[0])
        if source is None:
            raise ValueError("Could not read Blended Skill Talk train.json")
        rows = json.loads(source.read().decode("utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Blended Skill Talk train.json must contain a list")
    records = [
        converted for row in rows if isinstance(row, dict)
        for converted in [convert_blended_skill_talk_row(row)] if converted is not None
    ]
    random.Random(seed).shuffle(records)
    records = records[:limit]
    if not records:
        raise RuntimeError("Blended Skill Talk produced no usable examples")
    _write_jsonl(records, output)
    print(f"Prepared {len(records):,} Blended Skill Talk examples at {output}")
    return len(records)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare bounded CatAI SFT sources.")
    parser.add_argument("--source", choices=("smol-smoltalk", "blended-skill-talk"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.max_examples < 1 or args.seed < 0:
        raise ValueError("--max-examples must be positive and --seed non-negative")
    if args.source == "smol-smoltalk":
        prepare_smol_smoltalk(args.output, args.max_examples, args.seed)
    else:
        prepare_blended_skill_talk(args.output, args.max_examples, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
