"""Command-line interface for CatAI training."""

from __future__ import annotations

import argparse

import torch

from .batching import make_dataloader
from .checkpoint import save_checkpoint
from .config import TrainingConfig
from .corpus import tokenizer_and_tokens
from .dataset import TokenWindowDataset
from .device import resolve_device
from .model import CatAI
from .training import train_step


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


def main(argv: list[str] | None = None) -> int:
    """Train CatAI from a text corpus and save a checkpoint."""
    args = build_parser().parse_args(argv)
    if args.sequence_length < 1:
        raise ValueError("--sequence-length must be positive")
    if args.d_model < 1 or args.heads < 1 or args.layers < 1:
        raise ValueError("model dimensions must be positive")
    if args.d_model % args.heads != 0:
        raise ValueError("--d-model must be divisible by --heads")

    tokenizer, tokens = tokenizer_and_tokens(args.corpus)
    dataset = TokenWindowDataset(tokens, sequence_length=args.sequence_length)
    loader = make_dataloader(dataset, batch_size=args.batch_size, shuffle=True)
    device = resolve_device(args.device)
    model = CatAI(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=args.sequence_length,
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        grad_clip=args.grad_clip,
    )

    losses = []
    for epoch in range(config.epochs):
        total = 0.0
        batches = 0
        for batch in loader:
            total += train_step(model, optimizer, batch.to(device), grad_clip=config.grad_clip)
            batches += 1
        losses.append(total / batches)
        print(f"epoch {epoch + 1}/{config.epochs}: loss={losses[-1]:.4f}")

    save_checkpoint(
        args.checkpoint,
        model,
        optimizer,
        step=len(losses) * len(loader),
        epoch=len(losses),
        loss=losses[-1],
        metadata={
            "model": {
                "max_seq_len": args.sequence_length,
                "d_model": args.d_model,
                "n_heads": args.heads,
                "n_layers": args.layers,
            },
            "tokenizer": {
                "type": "char",
                "vocabulary": list(tokenizer.vocabulary),
            },
        },
    )
    print(f"saved checkpoint to {args.checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
