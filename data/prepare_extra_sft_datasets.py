#!/usr/bin/env python3
"""Prepare bounded Smol-SmolTalk and Blended Skill Talk SFT samples."""
from __future__ import annotations

import argparse
import email.utils
import hashlib
import io
import json
import random
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from catai.chat_dataset import default_state, validate_messages

SMOL_SMOLTALK_DATASET = "HuggingFaceTB/smol-smoltalk"
ULTRACHAT_DATASET = "HuggingFaceH4/ultrachat_200k"
GSM8K_DATASET = "openai/gsm8k"
BLENDED_SKILL_TALK_URL = "https://parl.ai/downloads/blended_skill_talk/blended_skill_talk.tar.gz"
BLENDED_SKILL_TALK_SHA256 = "5fbed0068ee89e2d43b93c3ecb341e784617033efa5e8e911a219d4eda6134a6"


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
    # The published dataset schema guarantees only messages and source.
    # Long examples are handled by CatAI's tokenizer-aware compact/skip logic
    # in ChatSupervisedDataset, rather than relying on undocumented columns.
    messages = normalize_messages(row.get("messages"))
    if messages is None:
        return None
    source = row.get("source")
    return {
        "state": default_state(),
        "messages": messages,
        "dataset_source": "smol-smoltalk",
        "dataset_license": "Apache-2.0",
        "dataset_repository": SMOL_SMOLTALK_DATASET,
        "upstream_source": source.strip() if isinstance(source, str) and source.strip() else "unknown",
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


def convert_ultrachat_row(row: dict[str, Any]) -> dict[str, Any] | None:
    messages = normalize_messages(row.get("messages"))
    if messages is None:
        return None
    return {
        "state": default_state(),
        "messages": messages,
        "dataset_source": "ultrachat_200k",
        "dataset_license": "MIT",
        "dataset_repository": ULTRACHAT_DATASET,
        "prompt_id": str(row.get("prompt_id") or ""),
    }


def convert_gsm8k_row(row: dict[str, Any]) -> dict[str, Any] | None:
    question, answer = row.get("question"), row.get("answer")
    if (
        not isinstance(question, str) or not question.strip()
        or not isinstance(answer, str) or not answer.strip()
    ):
        return None
    messages = [
        {
            "role": "user",
            "content": (
                "Solve this grade-school math problem. Show a clear, concise "
                "calculation and state the final answer.\n\n" + question.strip()
            ),
        },
        {"role": "assistant", "content": answer.strip()},
    ]
    try:
        messages = validate_messages(messages)
    except ValueError:
        return None
    return {
        "state": default_state(),
        "messages": messages,
        "dataset_source": "gsm8k",
        "dataset_license": "MIT",
        "dataset_repository": GSM8K_DATASET,
        "dataset_config": "main",
    }


DATASET_VIEWER_MAX_ATTEMPTS = 6
DATASET_VIEWER_PAGE_INTERVAL_SECONDS = 0.75
DATASET_VIEWER_MAX_CONSECUTIVE_RATE_LIMITED_PAGES = 3


class DatasetViewerRateLimitError(RuntimeError):
    """Raised when the Dataset Viewer keeps returning HTTP 429 for one page."""


def _dataset_viewer_retry_delay(retry_after: str | None, attempt: int) -> float:
    """Return a bounded delay, honoring Retry-After seconds or HTTP-date values."""
    delay: float | None = None
    if retry_after:
        try:
            delay = float(retry_after)
        except (TypeError, ValueError):
            try:
                retry_at = email.utils.parsedate_to_datetime(retry_after)
                delay = retry_at.timestamp() - time.time()
            except (TypeError, ValueError, OverflowError):
                delay = None
    if delay is None:
        delay = min(60.0, float(2 ** (attempt + 1)))
    return max(1.0, min(120.0, delay))


def _fetch_dataset_viewer_page(
    dataset: str, config: str, split: str, offset: int, length: int = 100
) -> dict[str, Any]:
    """Fetch one bounded page from the official Hugging Face Dataset Viewer API."""
    if offset < 0 or not 1 <= length <= 100:
        raise ValueError("Dataset Viewer offsets must be non-negative and page lengths must be 1..100")
    query = urllib.parse.urlencode({
        "dataset": dataset,
        "config": config,
        "split": split,
        "offset": offset,
        "length": length,
    })
    request = urllib.request.Request(
        f"https://datasets-server.huggingface.co/rows?{query}",
        headers={
            "Accept": "application/json",
            "User-Agent": "CatAI-SFT-dataset-preparation/1.0",
        },
    )
    last_error: BaseException | None = None
    for attempt in range(DATASET_VIEWER_MAX_ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            last_error = exc
            retryable = exc.code == 429 or 500 <= exc.code < 600
            if not retryable:
                raise RuntimeError(
                    f"Dataset Viewer returned HTTP {exc.code} for {dataset} at offset {offset}"
                ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
        if attempt + 1 < DATASET_VIEWER_MAX_ATTEMPTS:
            retry_after = (
                last_error.headers.get("Retry-After")
                if isinstance(last_error, urllib.error.HTTPError) and last_error.headers is not None
                else None
            )
            time.sleep(_dataset_viewer_retry_delay(retry_after, attempt))
            continue
        if isinstance(last_error, urllib.error.HTTPError) and last_error.code == 429:
            raise DatasetViewerRateLimitError(
                f"Hugging Face rate-limited {dataset} at offset {offset} "
                f"after {DATASET_VIEWER_MAX_ATTEMPTS} attempts"
            ) from last_error
        raise RuntimeError(
            f"Could not fetch {dataset} rows at offset {offset} "
            f"after {DATASET_VIEWER_MAX_ATTEMPTS} attempts"
        ) from last_error
    if not isinstance(payload, dict):
        raise ValueError(f"Dataset Viewer returned an invalid {dataset} response")
    if payload.get("partial") is True:
        raise RuntimeError(f"Dataset Viewer marked {dataset} as partial; refusing incomplete data")
    if not isinstance(payload.get("rows"), list):
        raise ValueError(f"Dataset Viewer returned an invalid rows page for {dataset}")
    return payload


def _sample_viewer_rows(
    fetch_page, limit: int, seed: int, dataset_label: str
) -> list[dict[str, Any]]:
    """Select deterministic examples from shuffled, bounded Dataset Viewer pages."""
    if limit < 1 or seed < 0:
        raise ValueError("limit must be positive and seed must be non-negative")
    first_page = fetch_page(0, 100)
    total_rows = first_page.get("num_rows_total")
    if not isinstance(total_rows, int) or total_rows < 0:
        raise ValueError(f"{dataset_label} Dataset Viewer omitted a valid num_rows_total")
    page_offsets = list(range(0, total_rows, 100))
    random.Random(seed).shuffle(page_offsets)
    pages_needed = (limit + 99) // 100
    max_pages = min(len(page_offsets), max(20, pages_needed * 5))
    selected: list[dict[str, Any]] = []
    last_request_at = time.monotonic()
    consecutive_rate_limited_pages = 0
    for offset in page_offsets[:max_pages]:
        if offset == 0:
            payload = first_page
            consecutive_rate_limited_pages = 0
        else:
            elapsed = time.monotonic() - last_request_at
            if elapsed < DATASET_VIEWER_PAGE_INTERVAL_SECONDS:
                time.sleep(DATASET_VIEWER_PAGE_INTERVAL_SECONDS - elapsed)
            try:
                payload = fetch_page(offset, min(100, total_rows - offset))
            except DatasetViewerRateLimitError as exc:
                consecutive_rate_limited_pages += 1
                last_request_at = time.monotonic()
                print(
                    f"Warning: skipping {dataset_label} page at offset {offset} "
                    f"after repeated Hugging Face HTTP 429 responses "
                    f"({consecutive_rate_limited_pages}/"
                    f"{DATASET_VIEWER_MAX_CONSECUTIVE_RATE_LIMITED_PAGES}).",
                    file=sys.stderr,
                )
                if consecutive_rate_limited_pages >= DATASET_VIEWER_MAX_CONSECUTIVE_RATE_LIMITED_PAGES:
                    raise RuntimeError(
                        f"{dataset_label} Dataset Viewer rate-limited "
                        f"{consecutive_rate_limited_pages} sampled pages in a row; "
                        "stopping rather than continuing against a throttled API"
                    ) from exc
                continue
            last_request_at = time.monotonic()
            consecutive_rate_limited_pages = 0
        if payload.get("partial") is True:
            raise RuntimeError(f"{dataset_label} Dataset Viewer returned a partial page")
        if payload.get("num_rows_total") != total_rows:
            raise RuntimeError(f"{dataset_label} row count changed while sampling")
        rows = payload.get("rows")
        if not isinstance(rows, list):
            raise ValueError(f"{dataset_label} Dataset Viewer returned an invalid page at offset {offset}")
        for entry in rows:
            if isinstance(entry, dict) and isinstance(entry.get("row"), dict):
                selected.append(entry["row"])
                if len(selected) >= limit:
                    break
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        raise RuntimeError(
            f"{dataset_label} produced only {len(selected):,} rows after checking "
            f"at most {max_pages:,} pages; requested {limit:,}"
        )
    random.Random(seed).shuffle(selected)
    return selected


def _fetch_ultrachat_page(offset: int, length: int = 100) -> dict[str, Any]:
    return _fetch_dataset_viewer_page(ULTRACHAT_DATASET, "default", "train_sft", offset, length)


def _fetch_gsm8k_page(offset: int, length: int = 100) -> dict[str, Any]:
    return _fetch_dataset_viewer_page(GSM8K_DATASET, "main", "train", offset, length)


def prepare_ultrachat(output: Path, limit: int, seed: int) -> int:
    rows = _sample_viewer_rows(_fetch_ultrachat_page, limit, seed, "UltraChat 200k")
    records = [
        converted for row in rows
        for converted in [convert_ultrachat_row(row)] if converted is not None
    ]
    if len(records) < limit:
        raise RuntimeError(f"UltraChat produced only {len(records):,} usable rows; requested {limit:,}")
    _write_jsonl(records[:limit], output)
    print(f"Prepared {limit:,} MIT-licensed UltraChat examples at {output}")
    return limit


def prepare_gsm8k(output: Path, limit: int, seed: int) -> int:
    rows = _sample_viewer_rows(_fetch_gsm8k_page, limit, seed, "GSM8K")
    records = [
        converted for row in rows
        for converted in [convert_gsm8k_row(row)] if converted is not None
    ]
    if len(records) < limit:
        raise RuntimeError(f"GSM8K produced only {len(records):,} usable rows; requested {limit:,}")
    _write_jsonl(records[:limit], output)
    print(f"Prepared {limit:,} MIT-licensed GSM8K examples at {output}")
    return limit


def _fetch_smol_smoltalk_page(offset: int, length: int = 100) -> dict[str, Any]:
    """Fetch a bounded Smol-SmolTalk page with shared Dataset Viewer retries."""
    return _fetch_dataset_viewer_page(
        SMOL_SMOLTALK_DATASET, "default", "train", offset, length
    )


def prepare_smol_smoltalk(output: Path, limit: int, seed: int) -> int:
    # Avoid datasets' remote Parquet streaming iterator. Breaking out of that
    # iterator can leave concurrent fsspec/Arrow reads alive during interpreter
    # shutdown, causing PyGILState_Release crashes in CI. The Dataset Viewer
    # serves bounded JSON pages (100 rows maximum) without local Arrow readers.
    first_page = _fetch_smol_smoltalk_page(0, 100)
    total_rows = first_page.get("num_rows_total")
    if not isinstance(total_rows, int) or total_rows < 0:
        raise ValueError("Hugging Face Dataset Viewer omitted a valid num_rows_total")
    if not isinstance(first_page.get("rows"), list):
        raise ValueError("Hugging Face Dataset Viewer returned an invalid rows page")

    page_offsets = list(range(0, total_rows, 100))
    random.Random(seed).shuffle(page_offsets)
    pages_needed = (limit + 99) // 100
    max_pages = min(len(page_offsets), max(20, pages_needed * 5))
    records: list[dict[str, Any]] = []

    for offset in page_offsets[:max_pages]:
        payload = first_page if offset == 0 else _fetch_smol_smoltalk_page(
            offset, min(100, total_rows - offset)
        )
        if payload.get("partial") is True:
            raise RuntimeError("Hugging Face Dataset Viewer returned a partial Smol-SmolTalk page")
        if payload.get("num_rows_total") != total_rows:
            raise RuntimeError("Smol-SmolTalk row count changed while pages were being sampled")
        rows = payload.get("rows")
        if not isinstance(rows, list):
            raise ValueError(f"Hugging Face Dataset Viewer returned an invalid page at offset {offset}")

        for entry in rows:
            if not isinstance(entry, dict):
                continue
            row = entry.get("row")
            if not isinstance(row, dict):
                continue
            item = convert_smol_smoltalk_row(row)
            if item is not None:
                records.append(item)
                if len(records) >= limit:
                    break
        if len(records) >= limit:
            break

    if not records:
        raise RuntimeError("Smol-SmolTalk produced no usable examples")
    if len(records) < limit:
        raise RuntimeError(
            f"Smol-SmolTalk produced only {len(records):,} usable examples after "
            f"checking at most {max_pages:,} pages; requested {limit:,}"
        )

    random.Random(seed).shuffle(records)
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
    parser.add_argument("--source", choices=("smol-smoltalk", "blended-skill-talk", "ultrachat", "gsm8k"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.max_examples < 1 or args.seed < 0:
        raise ValueError("--max-examples must be positive and --seed non-negative")
    if args.source == "smol-smoltalk":
        prepare_smol_smoltalk(args.output, args.max_examples, args.seed)
    elif args.source == "blended-skill-talk":
        prepare_blended_skill_talk(args.output, args.max_examples, args.seed)
    elif args.source == "ultrachat":
        prepare_ultrachat(args.output, args.max_examples, args.seed)
    else:
        prepare_gsm8k(args.output, args.max_examples, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
