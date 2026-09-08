"""Command-line interface for CatAI training."""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import random_split

from .batching import make_dataloader
from .checkpoint import save_checkpoint
from .config import TrainingConfig
from .corpus import tokenizer_and_tokens
from .dataset import TokenWindowDataset
from .device import resolve_device
from .model import CatAI
from .training import evaluate, train_step


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
    parser.add_argument("--validation-split", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=None, help="Early-stop after this many non-improving validation epochs.")
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
    config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        grad_clip=args.grad_clip,
        validation_split=args.validation_split,
        patience=args.patience,
    )

    tokenizer, tokens = tokenizer_and_tokens(args.corpus)
    dataset = TokenWindowDataset(tokens, sequence_length=args.sequence_length)
    validation_loader = None
    if config.validation_split > 0.0:
        validation_size = max(1, int(len(dataset) * config.validation_split))
        train_size = len(dataset) - validation_size
        if train_size < 1:
            raise ValueError("validation_split leaves no training samples")
        train_dataset, validation_dataset = random_split(
            dataset,
            [train_size, validation_size],
            generator=torch.Generator().manual_seed(42),
        )
    else:
        train_dataset = dataset
        validation_dataset = None

    loader = make_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True)
    if validation_dataset is not None:
        validation_loader = make_dataloader(validation_dataset, batch_size=args.batch_size, shuffle=False)

    device = resolve_device(args.device)
    model = CatAI(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=args.sequence_length,
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    losses: list[float] = []
    validation_losses: list[float] = []
    best_validation_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    total_batches = len(loader)
    progress_interval = max(1, total_batches // 10)

    for epoch in range(config.epochs):
        total = 0.0
        batches = 0
        print(f"epoch {epoch + 1}/{config.epochs}", flush=True)
        for batch_index, batch in enumerate(loader, start=1):
            total += train_step(model, optimizer, batch.to(device), grad_clip=config.grad_clip)
            batches += 1
            if batch_index % progress_interval == 0 or batch_index == total_batches:
                print(f"  batch {batch_index}/{total_batches}: loss={total / batches:.4f}", flush=True)
        epoch_loss = total / batches
        losses.append(epoch_loss)
        print(f"epoch {epoch + 1}/{config.epochs}: train_loss={epoch_loss:.4f}", flush=True)

        if validation_loader is not None:
            val_loss = evaluate(model, validation_loader, device=device)
            validation_losses.append(val_loss)
            print(f"epoch {epoch + 1}/{config.epochs}: val_loss={val_loss:.4f}", flush=True)
            if val_loss < best_validation_loss:
                best_validation_loss = val_loss
                best_epoch = epoch + 1
                stale_epochs = 0
                print("  new best validation loss", flush=True)
            else:
                stale_epochs += 1
                if config.patience is not None and stale_epochs >= config.patience:
                    print(f"early stopping at epoch {epoch + 1}", flush=True)
                    break

    final_epoch = len(losses)
    final_loss = losses[-1]
    save_checkpoint(
        args.checkpoint,
        model,
        optimizer,
        step=final_epoch * len(loader),
        epoch=final_epoch,
        loss=final_loss,
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
            "training": {
                "train_losses": losses,
                "validation_losses": validation_losses,
                "validation_split": config.validation_split,
                "patience": config.patience,
                "best_validation_loss": best_validation_loss if validation_loader is not None else None,
                "best_epoch": best_epoch if validation_loader is not None else None,
            },
        },
    )
    print(f"saved checkpoint to {args.checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
