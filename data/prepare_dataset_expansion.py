#!/usr/bin/env python3
"""Prepare the expanded Hugging Face chat and emotion dataset shortlist.

The script creates three auditable files:
- expanded_chat.jsonl: sources with a clear, reasonably permissive dataset license
- research_chat.jsonl: sources needing license/provenance review
- expanded_emotion.jsonl: emotion-label tasks, kept separate from conversational SFT

It samples bounded pages from the Hugging Face Dataset Viewer and records provenance
on every generated example. It never downloads full corpora into the Git repository.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from catai.chat_dataset import default_state, validate_messages

PAGE_SIZE = 100
MAX_ATTEMPTS = 6
PAGE_INTERVAL_SECONDS = 0.75
USER_AGENT = "CatAI-dataset-expansion/1.0"

CHAT_SOURCES: dict[str, dict[str, str]] = {
    "cat_v28": {
        "repository": "Nix-ai/Cat-v2.8", "config": "default", "split": "train",
        "license": "Apache-2.0", "bucket": "approved",
    },
    "catgirl_corpus": {
        "repository": "Compactbot/catgirl-1m-corpus", "config": "default", "split": "train",
        "license": "Apache-2.0", "bucket": "approved",
    },
    "anime_girl_chat": {
        "repository": "coderian/anime-girl-chat-dataset-en", "config": "default", "split": "train",
        "license": "CC-BY-4.0", "bucket": "approved",
    },
    "openhermes": {
        "repository": "teknium/OpenHermes-2.5", "config": "default", "split": "train",
        "license": "Unknown/mixed; no top-level license declared on the dataset card",
        "bucket": "research",
    },
    "ultrachat": {
        "repository": "HuggingFaceH4/ultrachat_200k", "config": "default", "split": "train_sft",
        "license": "MIT", "bucket": "approved",
    },
    "wizardlm": {
        "repository": "WizardLMTeam/WizardLM_evol_instruct_V2_196k", "config": "default", "split": "train",
        "license": "MIT (HF dataset card)", "bucket": "research",
    },
    "dolly": {
        "repository": "databricks/databricks-dolly-15k", "config": "default", "split": "train",
        "license": "CC-BY-SA-3.0", "bucket": "research",
    },
    "gpt4all": {
        "repository": "nomic-ai/gpt4all_prompt_generations", "config": "default", "split": "train",
        "license": "Apache-2.0 (HF dataset card)", "bucket": "approved",
    },
    "empathetic_dialogues": {
        "repository": "Estwld/empathetic_dialogues_llm", "config": "default", "split": "train",
        "license": "CC-BY-NC-4.0 (original upstream dataset)",
        "bucket": "research",
    },
    "esconv": {
        "repository": "Estwld/esconv_llm", "config": "default", "split": "train",
        "license": "Apache-2.0 (HF mirror card; upstream rights should be rechecked)",
        "bucket": "research",
    },
}

EMOTION_SOURCES: dict[str, dict[str, str]] = {
    "emotion": {
        "repository": "dair-ai/emotion", "config": "split", "split": "train",
        "license": "other/unspecified on the HF dataset card",
    },
    "many_emotions": {
        "repository": "ma2za/many_emotions", "config": "all", "split": "full",
        "license": "other/mixed; preserve the source-level license field on every row",
    },
    "meld": {
        "repository": "declare-lab/MELD", "config": "raw CSV", "split": "train",
        "license": "GPL-3.0; underlying Friends dialogue rights require separate review",
    },
}

EMOTION_LABELS_6 = ("sadness", "joy", "love", "anger", "fear", "surprise")
EMOTION_LABELS_7 = ("anger", "fear", "joy", "love", "sadness", "surprise", "neutral")
ROLE_ALIASES = {
    "human": "user", "prompter": "user", "usr": "user", "user": "user",
    "gpt": "assistant", "bot": "assistant", "sys": "assistant",
    "assistant": "assistant", "cat": "assistant", "anime_girl": "assistant",
    "system": "system",
}


def clean_text(value: object) -> str:
    return " ".join(value.split()) if isinstance(value, str) else ""


def normalize_messages(raw: object) -> list[dict[str, str]] | None:
    """Normalize common chat/ShareGPT formats, dropping source system prompts."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None
    if not isinstance(raw, list):
        return None

    messages: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        role = item.get("role", item.get("from", item.get("speaker")))
        content = item.get("content", item.get("value", item.get("text")))
        if not isinstance(role, str):
            continue
        role = ROLE_ALIASES.get(role.strip().lower(), role.strip().lower())
        content = clean_text(content)
        if role not in {"user", "assistant"} or not content:
            continue
        messages.append({"role": role, "content": content})

    while messages and messages[0]["role"] != "user":
        messages.pop(0)
    while messages and messages[-1]["role"] != "assistant":
        messages.pop()
    if len(messages) < 2:
        return None
    if any(left["role"] == right["role"] for left, right in zip(messages, messages[1:])):
        return None
    try:
        return validate_messages(messages)
    except ValueError:
        return None


def _chat_record(source: str, messages: list[dict[str, str]], spec: dict[str, str],
                 extra: dict[str, Any] | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "state": default_state(),
        "messages": messages,
        "dataset_source": source,
        "dataset_license": spec["license"],
        "dataset_repository": spec["repository"],
        "dataset_split": spec["split"],
    }
    if extra:
        record["dataset_metadata"] = extra
    return record


CAT_TURN_RE = re.compile(
    r"(?:^|\n)\s*User:\s*(.*?)\s*\n\s*Cat:\s*(.*?)(?=(?:\n\s*User:)|\Z)",
    flags=re.IGNORECASE | re.DOTALL,
)


def convert_chat_row(source: str, row: dict[str, Any],
                     spec: dict[str, str]) -> list[dict[str, Any]]:
    """Convert one HF row; catgirl-corpus rows can contain multiple turn pairs."""
    candidates: list[tuple[object, dict[str, Any] | None]] = []
    if source == "cat_v28":
        candidates.append((row.get("messages"), None))
    elif source == "catgirl_corpus":
        text = row.get("text")
        if isinstance(text, str):
            for user_text, assistant_text in CAT_TURN_RE.findall(text):
                candidates.append(([
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": assistant_text},
                ], None))
    elif source == "anime_girl_chat":
        candidates.append(([
            {"role": "user", "content": row.get("user", "")},
            {"role": "assistant", "content": row.get("anime_girl", "")},
        ], None))
    elif source in {"openhermes", "wizardlm"}:
        candidates.append((row.get("conversations"), {
            "source": row.get("source"), "category": row.get("category"),
        }))
    elif source == "ultrachat":
        candidates.append((row.get("messages"), {"prompt_id": row.get("prompt_id")}))
    elif source == "dolly":
        instruction = clean_text(row.get("instruction"))
        context = clean_text(row.get("context"))
        response = clean_text(row.get("response"))
        if context:
            instruction = instruction + "\n\nContext:\n" + context
        candidates.append(([
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ], {"category": row.get("category")}))
    elif source == "gpt4all":
        candidates.append(([
            {"role": "user", "content": row.get("prompt", "")},
            {"role": "assistant", "content": row.get("response", "")},
        ], {"source": row.get("source")}))
    elif source in {"empathetic_dialogues", "esconv"}:
        candidates.append((row.get("conversations"), {
            "emotion": row.get("emotion", row.get("emotion_type")),
            "situation": row.get("situation"),
            "problem": row.get("problem", row.get("problem_type")),
        }))
    else:
        raise ValueError(f"unsupported chat source: {source}")

    records: list[dict[str, Any]] = []
    for raw_messages, extra in candidates:
        messages = normalize_messages(raw_messages)
        if messages is not None:
            metadata = {key: value for key, value in (extra or {}).items() if value not in (None, "")}
            records.append(_chat_record(source, messages, spec, metadata))
    return records


def _viewer_request(repository: str, config: str, split: str,
                    offset: int, length: int) -> dict[str, Any]:
    if offset < 0 or not 1 <= length <= PAGE_SIZE:
        raise ValueError("Dataset Viewer offsets must be non-negative and page length must be 1..100")
    query = urllib.parse.urlencode({
        "dataset": repository, "config": config, "split": split,
        "offset": offset, "length": length,
    })
    request = urllib.request.Request(
        f"https://datasets-server.huggingface.co/rows?{query}",
        headers={"Accept": "application/json", "User-Agent": USER_AGENT},
    )
    last_error: BaseException | None = None
    payload: dict[str, Any] | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                loaded = json.loads(response.read().decode("utf-8"))
            if not isinstance(loaded, dict) or not isinstance(loaded.get("rows"), list):
                raise ValueError(f"invalid Dataset Viewer response for {repository}")
            payload = loaded
            break
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in {429, 500, 502, 503, 504}:
                raise RuntimeError(
                    f"Dataset Viewer returned HTTP {exc.code} for {repository}"
                ) from exc
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            try:
                delay = float(retry_after) if retry_after else float(2 ** (attempt + 1))
            except (TypeError, ValueError):
                delay = float(2 ** (attempt + 1))
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            last_error = exc
            delay = float(2 ** (attempt + 1))
        if attempt + 1 < MAX_ATTEMPTS:
            time.sleep(max(1.0, min(60.0, delay)))
    if payload is None:
        raise RuntimeError(
            f"could not fetch {repository} rows at offset {offset} after {MAX_ATTEMPTS} attempts"
        ) from last_error
    if payload.get("partial") is True:
        raise RuntimeError(f"Dataset Viewer marked {repository} data as partial")
    return payload


def sample_viewer_rows(repository: str, config: str, split: str,
                       limit: int, seed: int) -> list[dict[str, Any]]:
    """Select a deterministic bounded sample using shuffled 100-row pages."""
    if limit < 1 or seed < 0:
        raise ValueError("limit must be positive and seed must be non-negative")
    first_page = _viewer_request(repository, config, split, 0, PAGE_SIZE)
    total = first_page.get("num_rows_total")
    if not isinstance(total, int) or total < 0:
        raise ValueError(f"{repository} omitted a valid num_rows_total")
    if total == 0:
        return []
    offsets = list(range(0, total, PAGE_SIZE))
    random.Random(seed).shuffle(offsets)
    pages_needed = math.ceil(limit / PAGE_SIZE)
    max_pages = min(len(offsets), max(20, pages_needed * 5))
    selected: list[dict[str, Any]] = []
    last_request = 0.0
    for offset in offsets[:max_pages]:
        if offset == 0:
            payload = first_page
        else:
            elapsed = time.monotonic() - last_request
            if elapsed < PAGE_INTERVAL_SECONDS:
                time.sleep(PAGE_INTERVAL_SECONDS - elapsed)
            payload = _viewer_request(
                repository, config, split, offset, min(PAGE_SIZE, total - offset)
            )
            last_request = time.monotonic()
        if payload.get("num_rows_total") != total:
            raise RuntimeError(f"{repository} row count changed while sampling")
        for entry in payload["rows"]:
            if isinstance(entry, dict) and isinstance(entry.get("row"), dict):
                selected.append(entry["row"])
                if len(selected) >= limit:
                    break
        if len(selected) >= limit:
            break
    random.Random(seed).shuffle(selected)
    return selected


def label_to_string(raw: object, labels: tuple[str, ...]) -> str | None:
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int) and 0 <= raw < len(labels):
        return labels[raw]
    if isinstance(raw, str):
        value = raw.strip().lower()
        if value in labels:
            return value
    return None


def emotion_record(source: str, repository: str, license_name: str,
                   text: str, labels: list[str], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    instruction = (
        "Emotion-recognition exercise. Identify only the emotions explicitly supported "
        "by the message and its conversational context. Do not diagnose the speaker or "
        "invent hidden motives. Return the emotion label only."
    )
    record: dict[str, Any] = {
        "state": default_state(),
        "messages": [
            {"role": "system", "content": instruction},
            {"role": "user", "content": "Identify the emotion expressed:\n" + text},
            {"role": "assistant", "content": "Emotion: " + ", ".join(labels)},
        ],
        "dataset_source": source,
        "dataset_license": license_name,
        "dataset_repository": repository,
        "emotion_labels": labels,
    }
    if extra:
        record["dataset_metadata"] = extra
    return record


def convert_emotion_row(source: str, row: dict[str, Any],
                        spec: dict[str, str]) -> dict[str, Any] | None:
    text = clean_text(row.get("text"))
    if not text:
        return None
    if source == "emotion":
        label = label_to_string(row.get("label"), EMOTION_LABELS_6)
        if label is None:
            return None
        return emotion_record(
            source, spec["repository"], spec["license"], text, [label],
        )
    if source == "many_emotions":
        label = label_to_string(row.get("label"), EMOTION_LABELS_7)
        if label is None:
            return None
        origin = clean_text(row.get("dataset")) or "unknown"
        source_license = clean_text(row.get("license")) or "other/mixed"
        return emotion_record(
            source, spec["repository"], source_license, text, [label],
            {
                "upstream_dataset": origin,
                "language": row.get("language"),
                "source_id": row.get("source_id"),
                "corpus_note": "Many Emotions has mixed source-level licensing; review each row's provenance.",
            },
        )
    raise ValueError(f"unsupported row-based emotion source: {source}")


def _download_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read().decode("utf-8-sig")


def prepare_meld(limit: int, seed: int) -> list[dict[str, Any]]:
    base = "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/"
    text = _download_text(base + "train_sent_emo.csv")
    rows = list(csv.DictReader(io.StringIO(text)))
    rows.sort(key=lambda row: (
        int(row.get("Dialogue_ID", 0) or 0),
        int(row.get("Utterance_ID", 0) or 0),
    ))
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        dialog_id = str(row.get("Dialogue_ID", ""))
        utterance = clean_text(row.get("Utterance"))
        emotion = clean_text(row.get("Emotion")).lower()
        if dialog_id and utterance and emotion:
            grouped.setdefault(dialog_id, []).append({
                "utterance": utterance,
                "speaker": clean_text(row.get("Speaker")),
                "emotion": emotion,
            })
    examples: list[dict[str, Any]] = []
    license_name = EMOTION_SOURCES["meld"]["license"]
    repository = EMOTION_SOURCES["meld"]["repository"]
    for dialog_id, turns in grouped.items():
        for index, turn in enumerate(turns):
            context = turns[max(0, index - 2):index]
            context_lines = [
                f'{item["speaker"]}: {item["utterance"]}' for item in context
            ]
            prompt = "\n".join(context_lines + [f'{turn["speaker"]}: {turn["utterance"]}'])
            examples.append(emotion_record(
                "meld", repository, license_name, prompt, [turn["emotion"]],
                {"dialogue_id": dialog_id, "utterance_index": index, "context_turns": len(context)},
            ))
    random.Random(seed).shuffle(examples)
    return examples[:limit]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    count = 0
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                count += 1
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return count


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    approved_chat: list[dict[str, Any]] = []
    research_chat: list[dict[str, Any]] = []
    emotion_examples: list[dict[str, Any]] = []
    report: dict[str, Any] = {"chat": {}, "emotion": {}, "errors": []}

    for index, (source, spec) in enumerate(CHAT_SOURCES.items()):
        try:
            rows = sample_viewer_rows(
                spec["repository"], spec["config"], spec["split"],
                args.max_examples_per_source, args.seed + index,
            )
            converted = [
                item for row in rows
                for item in convert_chat_row(source, row, spec)
            ][:args.max_examples_per_source]
            if not converted:
                raise RuntimeError("no rows converted into valid user/assistant exchanges")
            destination = approved_chat if spec["bucket"] == "approved" else research_chat
            destination.extend(converted)
            report["chat"][source] = {
                "repository": spec["repository"], "license": spec["license"],
                "sampled_rows": len(rows), "usable_examples": len(converted),
                "bucket": spec["bucket"],
            }
            print(f"{source}: {len(converted):,} usable chat examples")
        except Exception as exc:
            report["errors"].append({"source": source, "error": f"{type(exc).__name__}: {exc}"})
            print(f"WARNING: {source} could not be prepared: {type(exc).__name__}: {exc}", file=sys.stderr)

    for index, (source, spec) in enumerate(EMOTION_SOURCES.items()):
        try:
            if source == "meld":
                converted = prepare_meld(args.max_examples_per_source, args.seed + 100 + index)
            else:
                rows = sample_viewer_rows(
                    spec["repository"], spec["config"], spec["split"],
                    args.max_examples_per_source, args.seed + 100 + index,
                )
                converted = [
                    item for row in rows
                    for item in [convert_emotion_row(source, row, spec)]
                    if item is not None
                ][:args.max_examples_per_source]
            if not converted:
                raise RuntimeError("no rows converted into emotion-label tasks")
            emotion_examples.extend(converted)
            report["emotion"][source] = {
                "repository": spec["repository"], "license": spec["license"],
                "usable_examples": len(converted),
            }
            print(f"{source}: {len(converted):,} emotion examples")
        except Exception as exc:
            report["errors"].append({"source": source, "error": f"{type(exc).__name__}: {exc}"})
            print(f"WARNING: {source} could not be prepared: {type(exc).__name__}: {exc}", file=sys.stderr)

    random.Random(args.seed).shuffle(approved_chat)
    random.Random(args.seed + 1).shuffle(research_chat)
    random.Random(args.seed + 2).shuffle(emotion_examples)
    counts = {
        "approved_chat": write_jsonl(args.chat_output, approved_chat),
        "research_chat": write_jsonl(args.research_chat_output, research_chat),
        "emotion": write_jsonl(args.emotion_output, emotion_examples),
    }
    report["outputs"] = {
        "chat": str(args.chat_output), "research_chat": str(args.research_chat_output),
        "emotion": str(args.emotion_output), "counts": counts,
    }
    report["sample_limit_per_source"] = args.max_examples_per_source
    report["seed"] = args.seed
    report["prepared_sources"] = len(report["chat"]) + len(report["emotion"])
    report["requested_sources"] = len(CHAT_SOURCES) + len(EMOTION_SOURCES)
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if counts["approved_chat"] + counts["research_chat"] + counts["emotion"] == 0:
        raise RuntimeError("no datasets produced usable examples; see the manifest for errors")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare bounded samples from CatAI's expanded Hugging Face dataset shortlist."
    )
    parser.add_argument("--max-examples-per-source", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chat-output", type=Path, default=Path("data/expanded_chat.jsonl"))
    parser.add_argument("--research-chat-output", type=Path, default=Path("data/research_chat.jsonl"))
    parser.add_argument("--emotion-output", type=Path, default=Path("data/expanded_emotion.jsonl"))
    parser.add_argument("--manifest-output", type=Path, default=Path("data/expanded_datasets_manifest.json"))
    args = parser.parse_args(argv)
    if args.max_examples_per_source < 1:
        raise ValueError("--max-examples-per-source must be positive")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    report = prepare(args)
    print(
        f"Prepared {report['outputs']['counts']['approved_chat']:,} approved-chat, "
        f"{report['outputs']['counts']['research_chat']:,} research-chat, and "
        f"{report['outputs']['counts']['emotion']:,} emotion examples."
    )
    print(
        f"Sources prepared: {report['prepared_sources']}/{report['requested_sources']}; "
        f"see {args.manifest_output} for row counts and any download/conversion errors."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
