#!/usr/bin/env python3
"""Download a small real-world public-domain corpus for CatAI experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlopen

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)


def download_tiny_shakespeare(output: Path) -> None:
    """Download Tiny Shakespeare as UTF-8 text."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(TINY_SHAKESPEARE_URL, timeout=30) as response:
        data = response.read()
    text = data.decode("utf-8")
    if not text.strip():
        raise RuntimeError("downloaded corpus is empty")
    output.write_text(text, encoding="utf-8")
    print(f"Downloaded {len(text):,} characters to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/tinyshakespeare.txt"))
    args = parser.parse_args()
    download_tiny_shakespeare(args.output)


if __name__ == "__main__":
    main()
