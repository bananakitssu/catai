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
from .training import train_epochs


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
    if args.grad_clip <= 0:
        raise ValueError("--grad-clip must be positive")

    tokenizer, tokens = tokenizer_and_tokens(args.corpus)
    dataset = TokenWindowDataset(tokens, sequence_length=args.sequence_length)
    loader = make_dataloader(dataset, batch_size=args.batch_size, shuffle=True)
    model = CatAI(
        vocab_size=len(tokenizer.vocab),
        max_seq_len=args.sequence_length,
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
    )
    device = resolve_device(args.device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        grad_clip=args.grad_clip,
    )

    losses = []
    for _ in range(config.epochs):
        total = 0.0
        batches = 0
        for batch in loader:
            batch = batch.to(device)
            total += __import__("catai.training", fromlist=["train_step"]).train_step(
                model, optimizer, batch, grad_clip=config.grad_clip
            )
            batches += 1
        losses.append(total / batches)
        print(f"epoch {len(losses)}/{config.epochs}: loss={losses[-1]:.4f}")

    save_checkpoint(
        args.checkpoint,
        model,
        optimizer,
        step=len(losses) * len(loader),
        epoch=len(losses),
        loss=losses[-1],
    )
    print(f"saved checkpoint to {args.checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
