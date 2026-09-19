#!/usr/bin/env python3
"""Download a configurable slice of RedPajama-V2 for CatAI experiments.

RedPajama-V2 is much larger than Tiny Shakespeare. This downloader deliberately
fetches only enough documents to reach ``--target-chars`` instead of attempting
to download the multi-terabyte dataset.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
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
REDPAJAMA_DEFAULT_MAX_DOCUMENTS = None


def _download(url: str, timeout: int = 60) -> bytes:
    """Download bytes with a simple identifying User-Agent."""
    request = Request(url, headers={"User-Agent": "CatAI-corpus-downloader/1.0"})
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def download_tiny_shakespeare(output: Path) -> None:
    """Download Tiny Shakespeare as UTF-8 text."""
    output.parent.mkdir(parents=True, exist_ok=True)
    data = _download(TINY_SHAKESPEARE_URL)
    text = data.decode("utf-8")
    if not text.strip():
        raise RuntimeError("downloaded corpus is empty")
    output.write_text(text, encoding="utf-8")
    print(f"Downloaded {len(text):,} characters to {output}")


def _extract_texts(payload: bytes) -> tuple[list[str], int]:
    """Extract usable text fields from JSONL records in a RedPajama shard."""
    raw = gzip.decompress(payload).decode("utf-8")
    texts: list[str] = []
    invalid_records = 0

    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            invalid_records += 1
            continue

        if not isinstance(record, dict):
            continue

        text = record.get("text")
        if isinstance(text, str) and text.strip():
            texts.append(text)

    return texts, invalid_records


def _print_progress(
    characters: int,
    target_chars: int,
    documents: int,
    processed_shards: int,
    total_shards: int,
    started_at: float,
    width: int = 28,
) -> None:
    """Print a CI-friendly progress bar on its own log line."""
    ratio = min(characters / target_chars, 1.0)
    filled = int(ratio * width)
    bar = "=" * filled + "-" * (width - filled)
    elapsed = max(time.monotonic() - started_at, 1e-9)
    rate = characters / elapsed
    remaining = max(target_chars - characters, 0)
    eta = remaining / rate if rate > 0 else None
    eta_text = f"{eta:.0f}s" if eta is not None else "n/a"

    print(
        f"[{bar}] {ratio * 100:6.2f}% | "
        f"{characters:,}/{target_chars:,} chars | "
        f"{documents:,} docs used | "
        f"{processed_shards:,}/{total_shards:,} shards | "
        f"{rate / 1_000_000:.2f}M chars/s | ETA {eta_text}"
    )


def download_redpajama(
    output: Path,
    target_chars: int,
    snapshot: str,
    language: str,
    partition: str,
    max_documents: int | None,
    seed: int,
) -> None:
    """Download RedPajama-V2 documents until the target size is reached.

    RedPajama publishes listings containing shard IDs. Each compressed shard
    contains many JSONL document records, so a shard is fetched and its records
    are consumed until the character/document target is reached.
    """
    if target_chars <= 0:
        raise ValueError("--target-chars must be greater than zero")
    if max_documents is not None and max_documents <= 0:
        raise ValueError("--max-documents must be greater than zero")

    output.parent.mkdir(parents=True, exist_ok=True)

    listings_name = f"{language}-{snapshot}-{partition}.txt"
    listings_url = f"{REDPAJAMA_BASE_URL}/listings/{listings_name}"
    print(f"Fetching RedPajama-V2 listing: {listings_url}")

    listing = _download(listings_url).decode("utf-8")
    shard_ids = [line.strip() for line in listing.splitlines() if line.strip()]
    if not shard_ids:
        raise RuntimeError("RedPajama listing was empty")

    # Deterministic sampling gives different parts of the huge listing while
    # keeping CI runs reproducible.
    random.Random(seed).shuffle(shard_ids)

    characters = 0
    documents = 0
    processed_shards = 0
    failures = 0
    started_at = time.monotonic()

    print(
        f"Starting corpus download: target={target_chars:,} characters, "
        f"shards={len(shard_ids):,}, "
        f"max-documents={max_documents or '∞'}"
    )

    with output.open("w", encoding="utf-8") as destination:
        for shard_id in shard_ids:
            if characters >= target_chars:
                break
            if max_documents is not None and documents >= max_documents:
                break

            processed_shards += 1
            shard_path = shard_id if shard_id.endswith(".json.gz") else f"{shard_id}.json.gz"
            url = f"{REDPAJAMA_BASE_URL}/documents/{shard_path}"
            try:
                texts, invalid_records = _extract_texts(_download(url))
                if invalid_records:
                    print(
                        f"Shard {shard_id}: skipped "
                        f"{invalid_records:,} malformed JSONL records"
                    )
            except Exception as exc:
                failures += 1
                if failures <= 10:
                    print(f"Skipping shard {shard_id}: {exc}")
                _print_progress(
                    characters,
                    target_chars,
                    documents,
                    processed_shards,
                    len(shard_ids),
                    started_at,
                )
                continue

            for text in texts:
                if characters >= target_chars:
                    break
                if max_documents is not None and documents >= max_documents:
                    break

                text = text.strip()
                if not text:
                    continue

                remaining = target_chars - characters
                chunk = text[:remaining]
                destination.write(chunk)
                destination.write("\n\n")
                characters += len(chunk)
                documents += 1

            _print_progress(
                characters,
                target_chars,
                documents,
                processed_shards,
                len(shard_ids),
                started_at,
            )

    if characters == 0:
        raise RuntimeError("RedPajama download produced no usable text")

    print(
        f"Downloaded {characters:,} characters from RedPajama-V2 "
        f"({documents:,} documents, {failures:,} skipped shards) to {output}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a CatAI training corpus")
    parser.add_argument(
        "--dataset",
        choices=("redpajama-v2", "tinyshakespeare"),
        default="redpajama-v2",
        help="Corpus source (default: redpajama-v2)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/corpus.txt"),
    )
    parser.add_argument(
        "--target-chars",
        type=int,
        default=100_000_000,
        help="Maximum characters to collect from RedPajama-V2 (default: 100M)",
    )
    parser.add_argument("--snapshot", default=REDPAJAMA_DEFAULT_SNAPSHOT)
    parser.add_argument("--language", default=REDPAJAMA_DEFAULT_LANGUAGE)
    parser.add_argument("--partition", default=REDPAJAMA_DEFAULT_PARTITION)
    parser.add_argument(
        "--max-documents",
        type=int,
        default=REDPAJAMA_DEFAULT_MAX_DOCUMENTS,
        help="Optional maximum number of documents to use (default: unlimited)",
    )
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    if args.dataset == "tinyshakespeare":
        download_tiny_shakespeare(args.output)
    else:
        download_redpajama(
            output=args.output,
            target_chars=args.target_chars,
            snapshot=args.snapshot,
            language=args.language,
            partition=args.partition,
            max_documents=args.max_documents,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
