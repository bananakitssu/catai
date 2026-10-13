#!/usr/bin/env python3
"""Export a CatAI training checkpoint to a Vercel-friendly NumPy bundle.

Training checkpoints keep optimizer state for resumability. The inference bundle
contains only model weights plus the exact model/tokenizer metadata required by
the lightweight NumPy runtime; no pickle objects are stored in the bundle.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def export_checkpoint(source: Path, destination: Path) -> None:
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint root must be a mapping")

    weights = checkpoint.get("model_state_dict")
    metadata = checkpoint.get("metadata")
    if not isinstance(weights, dict) or not weights:
        raise ValueError("checkpoint is missing model_state_dict")
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint is missing metadata")
    model = metadata.get("model")
    tokenizer = metadata.get("tokenizer")
    if not isinstance(model, dict) or not isinstance(tokenizer, dict):
        raise ValueError("checkpoint must contain model and tokenizer metadata")
    if not all(key in model for key in ("max_seq_len", "d_model", "n_heads", "n_layers")):
        raise ValueError("model metadata is incomplete")
    if not isinstance(tokenizer.get("vocabulary"), list) or tokenizer.get("type") not in ("char", "bpe"):
        raise ValueError("tokenizer metadata is incomplete")

    arrays: dict[str, np.ndarray] = {}
    for name, value in weights.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError("model_state_dict must contain named tensors only")
        # FP16 halves the deployed artifact size. The serving runtime computes
        # in FP32 to avoid accumulating low-precision attention errors.
        arrays[name] = value.detach().cpu().numpy().astype(np.float16, copy=False)

    if "token_embedding.weight" in arrays and "lm_head.weight" in arrays:
        if np.array_equal(arrays["token_embedding.weight"], arrays["lm_head.weight"]):
            # CatAI ties these weights; the loader restores the alias.
            del arrays["lm_head.weight"]

    bundle_metadata = {
        "format": "catai-numpy-inference",
        "format_version": 1,
        "step": int(checkpoint.get("step", 0)),
        "model": model,
        "tokenizer": tokenizer,
    }
    arrays["__metadata__"] = np.asarray(json.dumps(bundle_metadata, separators=(",", ":")))
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)

    # Reopen without pickle to validate the exact artifact being published.
    with np.load(destination, allow_pickle=False) as archive:
        loaded = json.loads(str(archive["__metadata__"].item()))
        if loaded.get("format") != "catai-numpy-inference" or "token_embedding.weight" not in archive:
            raise ValueError("exported inference bundle failed validation")
    print(f"Exported inference bundle: {destination} ({destination.stat().st_size:,} bytes)")
    print(f"Model context: {int(model['max_seq_len']):,} character/subword tokens")
    print(f"Tensor arrays: {len(arrays) - 1}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export_checkpoint(args.checkpoint, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
