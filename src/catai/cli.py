"""Command-line interface helpers for CatAI."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the CatAI training command."""
    parser = argparse.ArgumentParser(description="Train a small CatAI language model.")
    parser.add_argument("--corpus", required=True, help="Path to a UTF-8 text corpus.")
    parser.add_argument("--checkpoint", default="catai.pt", help="Checkpoint output path.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--device", default=None)
    return parser
