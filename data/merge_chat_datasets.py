#!/usr/bin/env python3
"""Command-line wrapper for preparing CatAI's merged OpenAssistant SFT data."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from catai.chat_merge import main


if __name__ == "__main__":
    raise SystemExit(main())
