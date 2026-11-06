#!/usr/bin/env python3
"""Build a filtered RedPajama-V2 + FineWeb-Edu corpus for CatAI."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import random
import shutil
import time
from pathlib import Path
from urllib.request import Request, urlopen

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)
REDPAJAMA_BASE_URL = "https://data.together.xyz/redpajama-data-v2/v1.0.0"
REDPAJAMA_DEFAULT_SNAPSHOT = "2023-06"
REDPAJAMA_DEFAULT_LANGUAGE = "en"
REDPAJAMA_DEFAULT_PARTITION = "head_middle"


def _download(url: str, timeout: int = 90) -> bytes:
    request = Request(url, headers={"User-Agent": "CatAI-corpus-downloader/2.0"})
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def download_tiny_shakespeare(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    text = _download(TINY_SHAKESPEARE_URL).decode("utf-8")
    if not text.strip():
        raise RuntimeError("downloaded Tiny Shakespeare corpus is empty")
    output.write_text(text, encoding="utf-8")


def _scores(signal_map: dict, name: str) -> list[float]:
    entries = signal_map.get(name)
    if not isinstance(entries, list):
        return []
    values = []
    for entry in entries:
        if isinstance(entry, list) and len(entry) >= 3:
            try:
                values.append(float(entry[2]))
            except (TypeError, ValueError):
                continue
    return values


def _document_score(signal_map: dict, name: str) -> float | None:
    values = _scores(signal_map, name)
    return sum(values) / len(values) if values else None


def gopher_rules_pass(signal_map: dict) -> bool:
    """Apply compact document-quality and repetition checks."""
    word_count = _document_score(signal_map, "rps_doc_word_count")
    mean_word_length = _document_score(signal_map, "rps_doc_mean_word_length")
    symbol_word_ratio = _document_score(signal_map, "rps_doc_symbol_to_word_ratio")
    top_2gram_fraction = _document_score(signal_map, "rps_doc_frac_chars_top_2gram")
    bullet_scores = _scores(signal_map, "rps_lines_start_with_bulletpoint")
    if any(value is None for value in (
        word_count, mean_word_length, symbol_word_ratio, top_2gram_fraction
    )):
        return False
    if not 50 <= word_count <= 100_000 or not 3 <= mean_word_length <= 10:
        return False
    if symbol_word_ratio > 0.1 or top_2gram_fraction > 0.2:
        return False
    if bullet_scores:
        fraction = sum(value >= 0.5 for value in bullet_scores) / len(bullet_scores)
        if fraction > 0.9:
            return False
    return True


def _duplicate_ids(payload: bytes) -> set[str]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError(
            "RedPajama duplicate filtering requires pyarrow; install with .[data]."
        ) from exc
    table = parquet.read_table(io.BytesIO(payload))
    columns = [
        name for name in ("id", "doc_id", "document_id", "id_int")
        if name in table.column_names
    ]
    if not columns:
        raise ValueError(f"Duplicate Parquet has no known ID column: {table.column_names!r}")
    result: set[str] = set()
    for name in columns:
        result.update(
            str(value) for value in table.column(name).to_pylist()
            if value is not None
        )
    return result


def filter_redpajama_shard(
    document_payload: bytes,
    signals_payload: bytes,
    shard_path: str,
    duplicate_ids: set[str],
    seen_digests: set[str],
    stats: dict[str, int] | None = None,
) -> list[str]:
    """Filter documents joined to quality-signal rows by their canonical IDs."""
    counters = stats if stats is not None else {}
    for key in (
        "seen", "duplicate_ids", "quality_rejected", "exact_duplicates", "kept",
        "missing_signals", "invalid_signal_records", "invalid_signal_ids", "duplicate_signal_ids",
    ):
        counters.setdefault(key, 0)
    documents = gzip.decompress(document_payload).decode("utf-8").splitlines()
    signal_lines = gzip.decompress(signals_payload).decode("utf-8").splitlines()

    # Quality-signal rows carry the document's canonical shard path and
    # zero-based row index in their "id" field. Join by that key instead of
    # assuming the two downloaded JSONL files always have identical lengths.
    # This also tolerates missing signal rows without shifting later documents.
    signal_prefix = f"{shard_path}/"
    signals_by_row: dict[int, dict] = {}
    for signal_line in signal_lines:
        try:
            signal_record = json.loads(signal_line)
        except json.JSONDecodeError:
            counters["invalid_signal_records"] += 1
            continue
        if not isinstance(signal_record, dict):
            counters["invalid_signal_records"] += 1
            continue
        signal_id = signal_record.get("id")
        if not isinstance(signal_id, str) or not signal_id.startswith(signal_prefix):
            counters["invalid_signal_ids"] += 1
            continue
        row_text = signal_id[len(signal_prefix):]
        if not row_text.isdecimal():
            counters["invalid_signal_ids"] += 1
            continue
        row_index = int(row_text)
        if row_index >= len(documents):
            counters["invalid_signal_ids"] = counters.get("invalid_signal_ids", 0) + 1
            continue
        if row_index in signals_by_row:
            counters["duplicate_signal_ids"] += 1
            continue
        signals_by_row[row_index] = signal_record

    selected = []
    for row_index, document_line in enumerate(documents):
        counters["seen"] += 1
        # RPv2 duplicate Parquet doc_id uses this same document path/index key.
        document_id = f"{shard_path}/{row_index}"
        if document_id in duplicate_ids:
            counters["duplicate_ids"] += 1
            continue
        signal_record = signals_by_row.get(row_index)
        if signal_record is None:
            counters["missing_signals"] += 1
            continue
        try:
            document = json.loads(document_line)
        except json.JSONDecodeError:
            counters["quality_rejected"] += 1
            continue
        if not isinstance(document, dict):
            counters["quality_rejected"] += 1
            continue
        signal_map, text = signal_record.get("quality_signals"), document.get("raw_content")
        if (
            not isinstance(signal_map, dict) or not isinstance(text, str)
            or not text.strip() or len(text.split()) < 10
            or not gopher_rules_pass(signal_map)
        ):
            counters["quality_rejected"] += 1
            continue
        normalized = text.strip()
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if digest in seen_digests:
            counters["exact_duplicates"] += 1
            continue
        seen_digests.add(digest)
        selected.append(normalized)
        counters["kept"] += 1
    return selected


def _progress(chars: int, target: int, docs: int, shard: int, total: int, started: float) -> None:
    fraction = min(chars / target, 1.0)
    elapsed = max(time.monotonic() - started, 1e-9)
    rate = chars / elapsed
    eta = max(target - chars, 0) / rate if rate else None
    eta_label = f"{eta:.0f}s" if eta is not None else "unknown"
    width = 28
    filled = int(fraction * width)
    print(
        f"[{'=' * filled}{'-' * (width-filled)}] {fraction*100:6.2f}% | "
        f"{chars:,}/{target:,} chars | {docs:,} docs | {shard:,}/{total:,} shards | ETA {eta_label}"
    )


def download_redpajama(
    output: Path, target_chars: int, snapshot: str, language: str,
    partition: str, max_documents: int | None, seed: int,
) -> dict[str, int]:
    """Download bounded RPv2 documents, applying quality filters and duplicate IDs."""
    if target_chars <= 0:
        raise ValueError("--target-chars must be positive")
    if max_documents is not None and max_documents <= 0:
        raise ValueError("--max-documents must be positive")
    if partition != "head_middle":
        raise ValueError("quality signals and duplicate ID artifacts are enabled for head_middle")
    try:
        import pyarrow.parquet  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("RedPajama filtering requires pyarrow; install .[data].") from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    listing_url = f"{REDPAJAMA_BASE_URL}/listings/{language}-{snapshot}-{partition}.txt"
    shard_ids = [
        line.strip() for line in _download(listing_url).decode("utf-8").splitlines()
        if line.strip()
    ]
    if not shard_ids:
        raise RuntimeError("RedPajama-V2 listing was empty")
    random.Random(seed).shuffle(shard_ids)
    chars = docs = processed = failures = 0
    stats: dict[str, int] = {}
    seen_digests: set[str] = set()
    started = time.monotonic()
    with output.open("w", encoding="utf-8") as destination:
        for shard_id in shard_ids:
            if chars >= target_chars or (max_documents is not None and docs >= max_documents):
                break
            processed += 1
            document_shard_path = (
                shard_id if shard_id.endswith(".json.gz") else f"{shard_id}.json.gz"
            )
            stem = document_shard_path.removesuffix(".json.gz")
            try:
                document_payload = _download(f"{REDPAJAMA_BASE_URL}/documents/{document_shard_path}")
                signal_payload = _download(f"{REDPAJAMA_BASE_URL}/quality_signals/{stem}.signals.json.gz")
                duplicate_payload = _download(f"{REDPAJAMA_BASE_URL}/duplicates/{stem}.duplicates.parquet")
                selected = filter_redpajama_shard(
                    document_payload, signal_payload, document_shard_path,
                    _duplicate_ids(duplicate_payload), seen_digests, stats,
                )
            except Exception as exc:
                failures += 1
                if failures <= 10:
                    print(f"Skipping shard {shard_id}: {type(exc).__name__}: {exc}")
                _progress(chars, target_chars, docs, processed, len(shard_ids), started)
                continue
            for text in selected:
                if chars >= target_chars or (max_documents is not None and docs >= max_documents):
                    break
                chunk = text[:target_chars - chars]
                if len(chunk.split()) < 10:
                    continue
                destination.write(chunk + "\n\n")
                chars += len(chunk)
                docs += 1
            _progress(chars, target_chars, docs, processed, len(shard_ids), started)
    if chars == 0:
        raise RuntimeError("RedPajama-V2 produced no usable text after filtering")
    print(
        f"Filtered RPv2: {chars:,} chars from {docs:,} docs; "
        f"{stats.get('duplicate_ids', 0):,} marked duplicates, "
        f"{stats.get('exact_duplicates', 0):,} exact duplicates, "
        f"{stats.get('quality_rejected', 0):,} rejected, "
        f"{stats.get('missing_signals', 0):,} docs without signals, "
        f"{stats.get('invalid_signal_ids', 0):,} invalid signal IDs, "
        f"{failures:,} failed shards."
    )
    return {"characters": chars, "documents": docs, "failed_shards": failures, **stats}


def download_fineweb_edu(
    output: Path, target_chars: int, seed: int,
    min_score: float = 3.0, min_language_score: float = 0.8,
) -> dict[str, int]:
    """Stream a bounded FineWeb-Edu sample selected by score and language score."""
    if target_chars <= 0:
        raise ValueError("--fineweb-edu-chars must be positive")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("FineWeb-Edu requires datasets; install .[data].") from exc
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu", split="train", streaming=True
    ).shuffle(seed=seed, buffer_size=10_000)
    chars = docs = rejected = 0
    with output.open("w", encoding="utf-8") as destination:
        for row in dataset:
            try:
                score = float(row.get("score", 0.0))
                language_score = float(row.get("language_score", 0.0))
            except (TypeError, ValueError):
                rejected += 1
                continue
            text = row.get("text")
            if (
                not isinstance(text, str) or not text.strip()
                or score < min_score or language_score < min_language_score
            ):
                rejected += 1
                continue
            chunk = text.strip()[:target_chars - chars]
            if not chunk:
                break
            destination.write(chunk + "\n\n")
            chars += len(chunk)
            docs += 1
            if chars >= target_chars:
                break
    if chars == 0:
        raise RuntimeError("FineWeb-Edu yielded no documents after filtering")
    print(f"FineWeb-Edu: {chars:,} chars from {docs:,} docs; {rejected:,} rows rejected.")
    return {"characters": chars, "documents": docs, "rejected": rejected}


def build_training_corpus(
    output: Path, *, target_chars: int, fineweb_edu_chars: int,
    snapshot: str, language: str, partition: str,
    max_documents: int | None, seed: int,
) -> None:
    """Write FineWeb-Edu first so it is included in prefix-capped training."""
    output.parent.mkdir(parents=True, exist_ok=True)
    rp_path = output.with_name(output.name + ".redpajama")
    edu_path = output.with_name(output.name + ".fineweb-edu")
    mixed_path = output.with_name(output.name + ".part")
    try:
        download_redpajama(
            rp_path, target_chars, snapshot, language, partition, max_documents, seed
        )
        if fineweb_edu_chars > 0:
            download_fineweb_edu(edu_path, fineweb_edu_chars, seed)
            with mixed_path.open("wb") as destination:
                with edu_path.open("rb") as source:
                    shutil.copyfileobj(source, destination)
                destination.write(b"\n\n")
                with rp_path.open("rb") as source:
                    shutil.copyfileobj(source, destination)
            mixed_path.replace(output)
        else:
            rp_path.replace(output)
    finally:
        rp_path.unlink(missing_ok=True)
        edu_path.unlink(missing_ok=True)
        mixed_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a curated CatAI corpus")
    parser.add_argument("--dataset", choices=("redpajama-v2", "tinyshakespeare"), default="redpajama-v2")
    parser.add_argument("--output", type=Path, default=Path("data/corpus.txt"))
    parser.add_argument("--target-chars", type=int, default=100_000_000)
    parser.add_argument("--fineweb-edu-chars", type=int, default=15_000_000)
    parser.add_argument("--snapshot", default=REDPAJAMA_DEFAULT_SNAPSHOT)
    parser.add_argument("--language", default=REDPAJAMA_DEFAULT_LANGUAGE)
    parser.add_argument("--partition", default=REDPAJAMA_DEFAULT_PARTITION)
    parser.add_argument("--max-documents", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    if args.dataset == "tinyshakespeare":
        download_tiny_shakespeare(args.output)
        return
    build_training_corpus(
        args.output, target_chars=args.target_chars, fineweb_edu_chars=args.fineweb_edu_chars,
        snapshot=args.snapshot, language=args.language, partition=args.partition,
        max_documents=args.max_documents, seed=args.seed,
    )


if __name__ == "__main__":
    main()
