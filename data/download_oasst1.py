#!/usr/bin/env python3
"""Generate CatAI chat JSONL from OpenAssistant/oasst1."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from catai.oasst1 import main


if __name__ == "__main__":
    raise SystemExit(main())
