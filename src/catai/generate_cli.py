"""Command-line interface for generating text from a CatAI checkpoint."""

from __future__ import annotations

import argparse

import torch

from .checkpoint import load_checkpoint, load_checkpoint_metadata
from .device import resolve_device
from .generation import generate
from .model import CatAI
from .tokenizer import CharTokenizer


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the CatAI generation command."""
    parser = argparse.ArgumentParser(description="Generate text with a CatAI checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to a CatAI checkpoint.")
    parser.add_argument("--prompt", required=True, help="Text prompt to continue.")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    parser.add_argument("--device", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Load a checkpoint and generate text from a prompt."""
    args = build_parser().parse_args(argv)
    if args.max_new_tokens < 0:
        raise ValueError("--max-new-tokens must be non-negative")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if args.repetition_penalty < 1.0:
        raise ValueError("--repetition-penalty must be at least 1.0")
    if args.no_repeat_ngram_size < 0:
        raise ValueError("--no-repeat-ngram-size must be non-negative")

    metadata = load_checkpoint_metadata(args.checkpoint)
    model_metadata = metadata.get("model")
    tokenizer_metadata = metadata.get("tokenizer")
    if not isinstance(model_metadata, dict) or not isinstance(tokenizer_metadata, dict):
        raise ValueError("checkpoint is missing model/tokenizer metadata")

    vocabulary = tokenizer_metadata.get("vocabulary")
    if tokenizer_metadata.get("type") != "char" or not isinstance(vocabulary, list):
        raise ValueError("checkpoint has unsupported tokenizer metadata")

    tokenizer = CharTokenizer(tuple(vocabulary))
    try:
        model = CatAI(
            vocab_size=tokenizer.vocab_size,
            max_seq_len=int(model_metadata["max_seq_len"]),
            d_model=int(model_metadata["d_model"]),
            n_heads=int(model_metadata["n_heads"]),
            n_layers=int(model_metadata["n_layers"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("checkpoint has invalid model metadata") from exc

    device = resolve_device(args.device)
    model.to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)

    prompt_tokens = tokenizer.encode(args.prompt)
    generated = generate(
        model,
        torch.tensor([prompt_tokens], dtype=torch.long, device=device),
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        eos_token_id=tokenizer.eos_token_id,
    )
    print(tokenizer.decode(generated[0].tolist()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
