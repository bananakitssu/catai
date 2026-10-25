#!/usr/bin/env python3
"""Download and convert the official OpenAssistant OASST2 ready tree export."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from catai.oasst1 import main as convert_openassistant_trees

OASST2_URL = (
    "https://huggingface.co/datasets/OpenAssistant/oasst2/resolve/main/"
    "2023-11-05_oasst2_ready.trees.jsonl.gz"
)


if __name__ == "__main__":
    # OASST1 and OASST2 share the same message-tree schema and quality filters.
    # The converter's defaults are overridden here; user flags can override them.
    defaults = [
        "--url", OASST2_URL,
        "--source", "data/oasst2_ready.trees.jsonl.gz",
        "--output", "data/oasst2_chat.jsonl",
    ]
    raise SystemExit(convert_openassistant_trees(defaults + sys.argv[1:]))
