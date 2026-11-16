#!/usr/bin/env python3
"""Build a provenance-tracked mixture of filtered CatAI pretraining corpora."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import random
import re
import shutil
import time
from pathlib import Path
from typing import TextIO
from urllib.request import Request, urlopen

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)
REDPAJAMA_BASE_URL = "https://data.together.xyz/redpajama-data-v2/v1.0.0"
REDPAJAMA_DEFAULT_SNAPSHOT = "2023-06"
REDPAJAMA_DEFAULT_LANGUAGE = "en"
REDPAJAMA_DEFAULT_PARTITION = "head_middle"
# Reserved pretraining document separator; tokenizers encode this as one EOS ID.
DOCUMENT_EOS_MARKER = "<EOS>"


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


def _write_document(destination: TextIO, text: str) -> None:
    """Write one document followed by the tokenizer's dedicated EOS marker."""
    destination.write(text.rstrip() + "\n" + DOCUMENT_EOS_MARKER + "\n\n")


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
                _write_document(destination, chunk)
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
            _write_document(destination, chunk)
            chars += len(chunk)
            docs += 1
            if chars >= target_chars:
                break
    if chars == 0:
        raise RuntimeError("FineWeb-Edu yielded no documents after filtering")
    print(f"FineWeb-Edu: {chars:,} chars from {docs:,} docs; {rejected:,} rows rejected.")
    return {"characters": chars, "documents": docs, "rejected": rejected}


COSMOPEDIA_REPOSITORY = "HuggingFaceTB/cosmopedia"
COSMOPEDIA_SUBSETS = (
    "openstax",
    "khanacademy",
    "stanford",
    "stories",
    "auto_math_text",
)
COMMON_CORPUS_REPOSITORY = "PleIAs/common_corpus"
COMMON_CORPUS_LICENSE_POLICY = (
    "Keep English documents only when the per-document license explicitly "
    "matches public domain, CC0, attribution-only CC BY, MIT, Apache-2.0, "
    "ODC-BY, or Licence Ouverte. Reject NC, ND, ShareAlike, unknown, and blank terms."
)


def common_corpus_license_allowed(value: object) -> bool:
    """Conservatively accept only explicit public-domain/permissive licenses."""
    if not isinstance(value, str) or not value.strip():
        return False
    license_text = re.sub(r"\s+", " ", value.strip()).upper()
    compact = re.sub(r"[^A-Z0-9]+", " ", license_text).strip()

    # These restrictions override any permissive-looking fragment.
    if any(term in compact for term in (
        "NON COMMERCIAL", "NONCOMMERCIAL", "NO DERIVATIVES",
        "SHARE ALIKE", "SHAREALIKE",
    )):
        return False
    if re.search(r"(?:^| )(?:NC|ND|SA)(?: |$)", compact):
        return False

    if "PUBLIC DOMAIN" in license_text or re.search(r"\bCC0(?:\b|[- ])", license_text):
        return True
    if re.search(r"\bCC[\s-]*BY(?:[\s-]*(?:2(?:\.0|\.5)?|3(?:\.0)?|4(?:\.0)?))?(?:\b|$)", license_text):
        return True
    if "CREATIVE COMMONS ATTRIBUTION" in license_text:
        return True
    if re.fullmatch(r"MIT(?: LICENSE)?", compact):
        return True
    if re.search(r"\bAPACHE(?: LICENSE)?[\s-]*2(?:\.0)?\b", license_text):
        return True
    if re.search(r"\bODC[\s-]*BY(?:[\s-]*1(?:\.0)?)?\b", license_text):
        return True
    if "LICENCE OUVERTE" in license_text or "OPEN GOVERNMENT LICENCE V3" in license_text:
        return True
    return False


def _write_provenance_record(destination: TextIO, record: dict[str, object]) -> None:
    destination.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def download_cosmopedia(
    output: Path,
    target_chars: int,
    seed: int,
) -> dict[str, object]:
    """Stream a bounded, deterministic mix of Cosmopedia's educational/story subsets."""
    if target_chars <= 0:
        raise ValueError("--cosmopedia-chars must be positive")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Cosmopedia requires datasets; install .[data].") from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    provenance_path = output.with_name(output.name + ".manifest.jsonl")
    chars = documents = rejected = duplicates = 0
    subset_stats: dict[str, dict[str, int]] = {}
    seen_digests: set[str] = set()
    per_subset_target = (target_chars + len(COSMOPEDIA_SUBSETS) - 1) // len(COSMOPEDIA_SUBSETS)

    with output.open("w", encoding="utf-8") as destination, provenance_path.open(
        "w", encoding="utf-8"
    ) as provenance:
        for subset_index, subset in enumerate(COSMOPEDIA_SUBSETS):
            subset_chars = subset_documents = subset_rejected = 0
            dataset = load_dataset(
                COSMOPEDIA_REPOSITORY,
                subset,
                split="train",
                streaming=True,
            ).shuffle(seed=seed + subset_index, buffer_size=1_000)
            for row in dataset:
                text = row.get("text") if isinstance(row, dict) else None
                if not isinstance(text, str) or len(text.strip()) < 100:
                    subset_rejected += 1
                    continue
                normalized = text.strip()
                digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
                if digest in seen_digests:
                    duplicates += 1
                    continue
                seen_digests.add(digest)
                remaining = min(
                    target_chars - chars,
                    per_subset_target - subset_chars,
                )
                if remaining <= 0:
                    break
                chunk = normalized[:remaining]
                if len(chunk.split()) < 20:
                    subset_rejected += 1
                    continue
                _write_document(destination, chunk)
                _write_provenance_record(provenance, {
                    "dataset": "cosmopedia",
                    "repository": COSMOPEDIA_REPOSITORY,
                    "license": "Apache-2.0 (dataset card)",
                    "subset": subset,
                    "seed_data": row.get("seed_data"),
                    "format": row.get("format"),
                    "audience": row.get("audience"),
                })
                chars += len(chunk)
                subset_chars += len(chunk)
                documents += 1
                subset_documents += 1
                if chars >= target_chars or subset_chars >= per_subset_target:
                    break
            subset_stats[subset] = {
                "characters": subset_chars,
                "documents": subset_documents,
                "rejected": subset_rejected,
            }
            rejected += subset_rejected
            if chars >= target_chars:
                break

    if chars == 0:
        raise RuntimeError("Cosmopedia yielded no usable text")
    print(
        f"Cosmopedia: {chars:,} chars from {documents:,} docs; "
        f"{duplicates:,} duplicates and {rejected:,} rejected."
    )
    return {
        "characters": chars,
        "documents": documents,
        "rejected": rejected,
        "duplicates": duplicates,
        "subsets": subset_stats,
        "license": "Apache-2.0 (publisher-declared; generated text uses external seed sources)",
    }


def download_common_corpus(
    output: Path,
    target_chars: int,
    seed: int,
    *,
    max_documents: int = 1_000,
) -> dict[str, object]:
    """Stream English Common Corpus text, accepting only explicitly allowed row licenses."""
    if target_chars <= 0:
        raise ValueError("--common-corpus-chars must be positive")
    if max_documents < 1:
        raise ValueError("--common-corpus-max-documents must be positive")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Common Corpus requires datasets; install .[data].") from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    provenance_path = output.with_name(output.name + ".manifest.jsonl")
    dataset = load_dataset(
        COMMON_CORPUS_REPOSITORY,
        split="train",
        streaming=True,
    ).shuffle(seed=seed, buffer_size=32)

    chars = documents = rejected_language = rejected_license = rejected_text = duplicates = 0
    license_counts: dict[str, int] = {}
    seen_digests: set[str] = set()
    with output.open("w", encoding="utf-8") as destination, provenance_path.open(
        "w", encoding="utf-8"
    ) as provenance:
        for row in dataset:
            if not isinstance(row, dict):
                rejected_text += 1
                continue
            language = str(row.get("language", "")).strip().lower()
            if language not in {"english", "en"}:
                rejected_language += 1
                continue
            license_name = row.get("license")
            if not common_corpus_license_allowed(license_name):
                rejected_license += 1
                continue
            text = row.get("text")
            if not isinstance(text, str):
                rejected_text += 1
                continue
            normalized = text.strip()
            if len(normalized.split()) < 50 or len(normalized) < 250:
                rejected_text += 1
                continue
            digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            if digest in seen_digests:
                duplicates += 1
                continue
            seen_digests.add(digest)
            remaining = target_chars - chars
            if remaining <= 0:
                break
            chunk = normalized[:remaining]
            if len(chunk.split()) < 30:
                rejected_text += 1
                continue
            _write_document(destination, chunk)
            _write_provenance_record(provenance, {
                "dataset": "common_corpus",
                "repository": COMMON_CORPUS_REPOSITORY,
                "identifier": row.get("identifier"),
                "collection": row.get("collection"),
                "open_type": row.get("open_type"),
                "license": license_name,
                "date": row.get("date"),
                "title": row.get("title"),
                "creator": row.get("creator"),
                "language": row.get("language"),
            })
            license_counts[str(license_name)] = license_counts.get(str(license_name), 0) + 1
            chars += len(chunk)
            documents += 1
            if chars >= target_chars or documents >= max_documents:
                break

    if chars == 0:
        raise RuntimeError(
            "Common Corpus produced no English documents with an explicitly approved license"
        )
    print(
        f"Common Corpus: {chars:,} chars from {documents:,} docs; "
        f"{rejected_language:,} non-English, {rejected_license:,} license-rejected, "
        f"{rejected_text:,} invalid and {duplicates:,} duplicate rows rejected."
    )
    return {
        "characters": chars,
        "documents": documents,
        "rejected_language": rejected_language,
        "rejected_license": rejected_license,
        "rejected_text": rejected_text,
        "duplicates": duplicates,
        "license_counts": license_counts,
        "license_policy": COMMON_CORPUS_LICENSE_POLICY,
    }


def _iter_corpus_documents(
    path: Path,
    provenance_path: Path | None,
    fallback_metadata: dict[str, object],
):
    """Yield document text paired with its sidecar provenance (or source defaults)."""
    metadata_stream = (
        provenance_path.open("r", encoding="utf-8")
        if provenance_path is not None and provenance_path.is_file()
        else None
    )
    try:
        with path.open("r", encoding="utf-8") as source:
            document_lines: list[str] = []
            for line in source:
                if line.rstrip("\r\n") == DOCUMENT_EOS_MARKER:
                    document = "".join(document_lines).strip()
                    document_lines = []
                    if not document:
                        continue
                    metadata = dict(fallback_metadata)
                    if metadata_stream is not None:
                        metadata_line = metadata_stream.readline()
                        if not metadata_line:
                            raise ValueError(
                                f"provenance sidecar ended before corpus documents in {path}"
                            )
                        parsed = json.loads(metadata_line)
                        if not isinstance(parsed, dict):
                            raise ValueError(f"invalid provenance record in {provenance_path}")
                        metadata.update(parsed)
                    yield document, metadata
                else:
                    document_lines.append(line)
            trailing = "".join(document_lines).strip()
            if trailing:
                metadata = dict(fallback_metadata)
                if metadata_stream is not None:
                    metadata_line = metadata_stream.readline()
                    if not metadata_line:
                        raise ValueError(
                            f"provenance sidecar ended before trailing corpus document in {path}"
                        )
                    parsed = json.loads(metadata_line)
                    if not isinstance(parsed, dict):
                        raise ValueError(f"invalid provenance record in {provenance_path}")
                    metadata.update(parsed)
                yield trailing, metadata
            if metadata_stream is not None and metadata_stream.readline():
                raise ValueError(
                    f"provenance sidecar has more records than corpus documents in {provenance_path}"
                )
    finally:
        if metadata_stream is not None:
            metadata_stream.close()


def mix_corpus_files(
    output: Path,
    provenance_output: Path,
    sources: list[tuple[Path, Path | None, dict[str, object]]],
) -> dict[str, int]:
    """Round-robin source documents so prefix-capped training sees every corpus."""
    output.parent.mkdir(parents=True, exist_ok=True)
    provenance_output.parent.mkdir(parents=True, exist_ok=True)
    iterators = [
        iter(_iter_corpus_documents(path, sidecar, metadata))
        for path, sidecar, metadata in sources
    ]
    active = [True] * len(iterators)
    counts: dict[str, int] = {}
    with output.open("w", encoding="utf-8") as destination, provenance_output.open(
        "w", encoding="utf-8"
    ) as provenance:
        while any(active):
            for index, iterator in enumerate(iterators):
                if not active[index]:
                    continue
                try:
                    document, metadata = next(iterator)
                except StopIteration:
                    active[index] = False
                    continue
                _write_document(destination, document)
                _write_provenance_record(provenance, metadata)
                source_name = str(metadata.get("dataset", "unknown"))
                counts[source_name] = counts.get(source_name, 0) + 1
    if not any(counts.values()):
        raise RuntimeError("mixed corpus contains no documents")
    return counts


def build_training_corpus(
    output: Path, *, target_chars: int, fineweb_edu_chars: int,
    snapshot: str, language: str, partition: str,
    max_documents: int | None, seed: int,
    cosmopedia_chars: int = 100_000_000,
    common_corpus_chars: int = 100_000_000,
    common_corpus_max_documents: int = 1_000,
) -> None:
    """Build a round-robin mix of quality-filtered RedPajama, FineWeb, and new sources."""
    if fineweb_edu_chars < 0 or cosmopedia_chars < 0 or common_corpus_chars < 0:
        raise ValueError("corpus source character budgets must be non-negative")
    output.parent.mkdir(parents=True, exist_ok=True)
    rp_path = output.with_name(output.name + ".redpajama")
    edu_path = output.with_name(output.name + ".fineweb-edu")
    cosmopedia_path = output.with_name(output.name + ".cosmopedia")
    common_path = output.with_name(output.name + ".common-corpus")
    mixed_manifest = output.with_name(output.name + ".provenance.jsonl")
    sources: list[tuple[Path, Path | None, dict[str, object]]] = []
    summary: dict[str, object] = {
        "format": "catai-base-corpus-manifest",
        "version": 1,
        "seed": seed,
        "sources": {},
        "provenance_file": str(mixed_manifest),
        "common_corpus_license_policy": COMMON_CORPUS_LICENSE_POLICY,
    }
    temporary_paths = [rp_path, edu_path, cosmopedia_path, common_path]
    try:
        rp_stats = download_redpajama(
            rp_path, target_chars, snapshot, language, partition, max_documents, seed
        )
        sources.append((rp_path, None, {
            "dataset": "redpajama-v2",
            "repository": "togethercomputer/RedPajama-Data-V2",
            "license": "Common Crawl Terms of Use; see dataset card",
        }))
        summary["sources"]["redpajama-v2"] = rp_stats

        if fineweb_edu_chars > 0:
            edu_stats = download_fineweb_edu(edu_path, fineweb_edu_chars, seed)
            sources.append((edu_path, None, {
                "dataset": "fineweb-edu",
                "repository": "HuggingFaceFW/fineweb-edu",
                "license": "ODC-BY-1.0; Common Crawl and source terms also apply",
            }))
            summary["sources"]["fineweb-edu"] = edu_stats

        if cosmopedia_chars > 0:
            cosmopedia_stats = download_cosmopedia(cosmopedia_path, cosmopedia_chars, seed)
            sources.append((
                cosmopedia_path,
                cosmopedia_path.with_name(cosmopedia_path.name + ".manifest.jsonl"),
                {
                    "dataset": "cosmopedia",
                    "repository": COSMOPEDIA_REPOSITORY,
                    "license": "Apache-2.0 (dataset card)",
                },
            ))
            summary["sources"]["cosmopedia"] = cosmopedia_stats
            temporary_paths.append(
                cosmopedia_path.with_name(cosmopedia_path.name + ".manifest.jsonl")
            )

        if common_corpus_chars > 0:
            common_stats = download_common_corpus(
                common_path,
                common_corpus_chars,
                seed,
                max_documents=common_corpus_max_documents,
            )
            sources.append((
                common_path,
                common_path.with_name(common_path.name + ".manifest.jsonl"),
                {
                    "dataset": "common_corpus",
                    "repository": COMMON_CORPUS_REPOSITORY,
                    "license": "Per-document; see provenance sidecar",
                },
            ))
            summary["sources"]["common-corpus"] = common_stats
            temporary_paths.append(
                common_path.with_name(common_path.name + ".manifest.jsonl")
            )

        document_counts = mix_corpus_files(output, mixed_manifest, sources)
        summary["mixed_document_counts"] = document_counts
        summary["total_documents"] = sum(document_counts.values())
        summary["total_source_chars"] = sum(
            int(item.get("characters", 0))
            for item in summary["sources"].values()
            if isinstance(item, dict)
        )
        manifest_path = output.with_name(output.name + ".manifest.json")
        manifest_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(
            f"Mixed corpus: {summary['total_documents']:,} documents from "
            f"{len(document_counts)} sources; provenance: {mixed_manifest}"
        )
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a curated CatAI corpus")
    parser.add_argument("--dataset", choices=("redpajama-v2", "tinyshakespeare"), default="redpajama-v2")
    parser.add_argument("--output", type=Path, default=Path("data/corpus.txt"))
    parser.add_argument("--target-chars", type=int, default=100_000_000)
    parser.add_argument("--fineweb-edu-chars", type=int, default=15_000_000)
    parser.add_argument("--cosmopedia-chars", type=int, default=100_000_000)
    parser.add_argument("--common-corpus-chars", type=int, default=100_000_000)
    parser.add_argument("--common-corpus-max-documents", type=int, default=1_000)
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
        args.output,
        target_chars=args.target_chars,
        fineweb_edu_chars=args.fineweb_edu_chars,
        cosmopedia_chars=args.cosmopedia_chars,
        common_corpus_chars=args.common_corpus_chars,
        common_corpus_max_documents=args.common_corpus_max_documents,
        snapshot=args.snapshot,
        language=args.language,
        partition=args.partition,
        max_documents=args.max_documents,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
