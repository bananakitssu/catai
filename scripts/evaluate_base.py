#!/usr/bin/env python3
"""Report base-checkpoint metrics and run deterministic generation sanity checks."""

from __future__ import annotations

import argparse
import math

import torch

from catai.checkpoint import load_checkpoint, load_checkpoint_metadata
from catai.device import resolve_device
from catai.generate_cli import _load_tokenizer
from catai.generation import generate
from catai.model import CatAI


PROMPTS = (
    "The cat is",
    "A computer program uses variables to",
    "Language models learn patterns from",
    "One way to solve this problem is",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-unk-fraction", type=float, default=0.10)
    parser.add_argument("--device", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if args.top_k < 1:
        raise ValueError("--top-k must be positive")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if not 0.0 <= args.max_unk_fraction <= 1.0:
        raise ValueError("--max-unk-fraction must be in [0, 1]")

    metadata = load_checkpoint_metadata(args.checkpoint)
    model_metadata = metadata.get("model")
    tokenizer_metadata = metadata.get("tokenizer")
    training = metadata.get("training")
    if not isinstance(model_metadata, dict) or not isinstance(tokenizer_metadata, dict):
        raise ValueError("checkpoint is missing model/tokenizer metadata")
    if not isinstance(training, dict):
        raise ValueError("checkpoint is missing base-training diagnostics")

    train_losses = training.get("train_losses")
    validation_losses = training.get("validation_losses")
    if not isinstance(train_losses, list) or not train_losses:
        raise ValueError("checkpoint has no recorded training losses")
    if not isinstance(validation_losses, list) or not validation_losses:
        raise ValueError("checkpoint has no recorded validation losses")
    train_loss = float(train_losses[-1])
    validation_loss = float(validation_losses[-1])
    if not math.isfinite(train_loss) or not math.isfinite(validation_loss):
        raise ValueError("training or validation loss is not finite")

    tokenizer = _load_tokenizer(tokenizer_metadata)
    parameter_count = int(model_metadata.get("parameter_count", 0))
    token_count = int(training.get("total_tokens", 0))
    unk_fraction = float(training.get("unk_token_fraction", float("nan")))
    if parameter_count < 1 or token_count < 1:
        raise ValueError("checkpoint has invalid parameter/token counts")
    if not math.isfinite(unk_fraction) or not 0.0 <= unk_fraction <= 1.0:
        raise ValueError("checkpoint has no valid tokenizer UNK-frequency diagnostic")
    if unk_fraction > args.max_unk_fraction:
        raise SystemExit(
            f"Base checkpoint UNK fraction {unk_fraction:.2%} exceeds "
            f"the configured limit {args.max_unk_fraction:.2%}."
        )

    random_baseline = math.log(tokenizer.vocab_size)
    print(f"Parameters: {parameter_count:,}")
    print(f"Training tokens: {token_count:,}")
    print(f"Tokens/parameter: {token_count / parameter_count:.2f}")
    print(f"Training loss: {train_loss:.4f}")
    print(f"Validation loss: {validation_loss:.4f} (uniform-random baseline: {random_baseline:.4f})")
    print(f"Tokenizer UNK tokens: {int(training.get('unk_token_count', 0)):,} ({unk_fraction:.2%})")
    if validation_loss >= random_baseline:
        raise SystemExit(
            "Base checkpoint did not beat the uniform-random next-token loss baseline."
        )

    model = CatAI(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=int(model_metadata["max_seq_len"]),
        d_model=int(model_metadata["d_model"]),
        n_heads=int(model_metadata["n_heads"]),
        n_layers=int(model_metadata["n_layers"]),
    )
    device = resolve_device(args.device)
    model.to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    torch.manual_seed(args.seed)

    failed_checks: list[str] = []
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    for prompt in PROMPTS:
        prompt_tokens = tokenizer.encode(prompt)
        output_ids = generate(
            model,
            torch.tensor([prompt_tokens], dtype=torch.long, device=device),
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=min(args.top_k, tokenizer.vocab_size),
            eos_token_id=tokenizer.eos_token_id,
        )
        generated_ids = output_ids[0, len(prompt_tokens):].detach().cpu().tolist()
        reply = tokenizer.decode(generated_ids).strip()
        visible_ids = [token for token in generated_ids if token != tokenizer.eos_token_id]
        generated_unk_fraction = (
            sum(token == unk_token_id for token in visible_ids) / len(visible_ids)
            if visible_ids else 0.0
        )
        print(f"\nPrompt: {prompt}")
        print(f"Completion: {reply or '[empty completion]'}")
        print(
            f"Generated tokens: {len(visible_ids)}; "
            f"UNK fraction: {generated_unk_fraction:.2%}",
            flush=True,
        )
        if not reply:
            failed_checks.append(f"{prompt!r}: empty completion")
        if generated_unk_fraction > args.max_unk_fraction:
            failed_checks.append(
                f"{prompt!r}: generated UNK fraction {generated_unk_fraction:.2%} "
                f"exceeds {args.max_unk_fraction:.2%}"
            )

    if failed_checks:
        raise SystemExit("Base generation sanity checks failed: " + "; ".join(failed_checks))
    print("Base checkpoint sanity checks passed. Generated text still needs human semantic review.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
