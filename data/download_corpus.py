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
REDPAJAMA_DEFAULT_MAX_DOCUMENTS = 20


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


def _extract_text(payload: bytes) -> str:
    """Extract document text from a RedPajama-V2 gzip-compressed JSON file."""
    raw = gzip.decompress(payload).decode("utf-8")
    record = json.loads(raw)

    if isinstance(record, dict):
        text = record.get("text")
        if isinstance(text, str):
            return text

    raise ValueError("RedPajama document did not contain a string 'text' field")


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

    RedPajama publishes listings containing document IDs. Individual compressed
    documents are fetched only until enough text has been collected, so this
    does not attempt to download an entire ~1 TB snapshot.
    """
    if target_chars <= 0:
        raise ValueError("--target-chars must be greater than zero")

    output.parent.mkdir(parents=True, exist_ok=True)

    listings_name = f"{language}-{snapshot}-{partition}.txt"
    listings_url = f"{REDPAJAMA_BASE_URL}/listings/{listings_name}"
    print(f"Fetching RedPajama-V2 listing: {listings_url}")

    listing = _download(listings_url).decode("utf-8")
    document_ids = [line.strip() for line in listing.splitlines() if line.strip()]
    if not document_ids:
        raise RuntimeError("RedPajama listing was empty")

    # Deterministic sampling gives different parts of the huge listing while
    # keeping CI runs reproducible.
    random.Random(seed).shuffle(document_ids)

    characters = 0
    documents = 0
    failures = 0

    with output.open("w", encoding="utf-8") as destination:
        for document_id in document_ids:
            if characters >= target_chars:
                break
            if max_documents is not None and documents >= max_documents:
                break

            url = f"{REDPAJAMA_BASE_URL}/documents/{document_id}.json.gz"
            try:
                text = _extract_text(_download(url))
            except Exception as exc:  # Keep one bad web document from killing a run.
                failures += 1
                if failures <= 10:
                    print(f"Skipping document {document_id}: {exc}")
                continue

            text = text.strip()
            if not text:
                continue

            remaining = target_chars - characters
            destination.write(text[:remaining])
            destination.write("\n\n")
            characters += min(len(text), remaining)
            documents += 1

            print(
                f"Collected {characters:,}/{target_chars:,} characters "
                f"from {documents:,}/{max_documents or '∞'} documents"
            )

    if characters == 0:
        raise RuntimeError("RedPajama download produced no usable text")

    print(
        f"Downloaded {characters:,} characters from RedPajama-V2 "
        f"({documents:,} documents, {failures:,} skipped) to {output}"
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
        help="Maximum number of documents to download (default: 20)",
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
