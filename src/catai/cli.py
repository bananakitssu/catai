"""Command-line interface for CatAI training."""

from __future__ import annotations

import argparse
import time

import torch
from torch.utils.data import random_split

from .batching import make_dataloader
from .checkpoint import save_checkpoint
from .config import MODEL_PRESETS, ModelConfig, TrainingConfig
from .corpus import tokenizer_and_tokens
from .dataset import TokenWindowDataset
from .device import resolve_device
from .model import CatAI
from .training import evaluate, train_step


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the CatAI training command."""
    parser = argparse.ArgumentParser(description="Train a CatAI language model.")
    parser.add_argument("--corpus", required=True, help="Path to a UTF-8 text corpus.")
    parser.add_argument("--checkpoint", default="catai.pt", help="Checkpoint output path.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    # Raised from 32 → 256. Character-level models need more room to form words.
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument(
        "--window-stride",
        type=int,
        default=None,
        help="Token stride between training windows (default: sequence length; use 1 for fully overlapping windows).",
    )
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--model-size", choices=tuple(MODEL_PRESETS), default="tiny", help="Use a larger model preset.")
    parser.add_argument("--tokenizer", choices=("char", "bpe"), default="char")
    parser.add_argument("--vocab-size", type=int, default=256, help="Target vocabulary size for BPE.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Cap the training token budget.")
    parser.add_argument("--validation-split", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=None, help="Early-stop after this many non-improving validation epochs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible training.")
    parser.add_argument("--device", default=None)
    return parser


def _progress_bar(completed: int, total: int, width: int = 30) -> str:
    """Return a compact textual progress bar."""
    if total <= 0:
        return "[" + "-" * width + "]"
    fraction = min(1.0, max(0.0, completed / total))
    filled = int(width * fraction)
    return "[" + "=" * filled + ">" + "-" * max(0, width - filled - 1) + "]"


def main(argv: list[str] | None = None) -> int:
    """Train CatAI from a text corpus and save a checkpoint."""
    args = build_parser().parse_args(argv)
    if args.sequence_length < 1:
        raise ValueError("--sequence-length must be positive")
    if args.window_stride is not None and args.window_stride < 1:
        raise ValueError("--window-stride must be positive")
    if args.d_model < 1 or args.heads < 1 or args.layers < 1:
        raise ValueError("model dimensions must be positive")
    if args.d_model % args.heads != 0:
        raise ValueError("--d-model must be divisible by --heads")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if args.vocab_size < 2:
        raise ValueError("--vocab-size must be at least 2")

    model_config = ModelConfig.from_preset(args.model_size)
    if args.model_size != "tiny":
        d_model, n_heads, n_layers = model_config.d_model, model_config.n_heads, model_config.n_layers
    else:
        d_model, n_heads, n_layers = args.d_model, args.heads, args.layers

    config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        grad_clip=args.grad_clip,
        validation_split=args.validation_split,
        patience=args.patience,
        max_tokens=args.max_tokens,
    )
    torch.manual_seed(args.seed)

    tokenizer, tokens = tokenizer_and_tokens(
        args.corpus,
        tokenizer_type=args.tokenizer,
        vocab_size=args.vocab_size,
        max_tokens=config.max_tokens,
    )
    window_stride = args.window_stride if args.window_stride is not None else args.sequence_length
    dataset = TokenWindowDataset(
        tokens,
        sequence_length=args.sequence_length,
        stride=window_stride,
    )
    validation_loader = None
    if config.validation_split > 0.0:
        validation_size = max(1, int(len(dataset) * config.validation_split))
        train_size = len(dataset) - validation_size
        if train_size < 1:
            raise ValueError("validation_split leaves no training samples")
        train_dataset, validation_dataset = random_split(
            dataset,
            [train_size, validation_size],
            generator=torch.Generator().manual_seed(args.seed),
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
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    losses: list[float] = []
    validation_losses: list[float] = []
    best_validation_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    total_batches = len(loader)
    progress_interval = max(1, total_batches // 10)
    total_steps = 0
    total_tokens = 0
    started_at = time.perf_counter()

    for epoch in range(config.epochs):
        total = 0.0
        batches = 0
        print(f"epoch {epoch + 1}/{config.epochs}", flush=True)
        for batch_index, batch in enumerate(loader, start=1):
            batch = batch.to(device)
            total += train_step(model, optimizer, batch, grad_clip=config.grad_clip)
            batches += 1
            total_steps += 1
            total_tokens += batch.numel() - batch.size(0)
            if batch_index % progress_interval == 0 or batch_index == total_batches:
                elapsed = max(time.perf_counter() - started_at, 1e-9)
                tokens_per_second = total_tokens / elapsed
                completed_batches = (epoch * total_batches) + batch_index
                total_training_batches = config.epochs * total_batches
                remaining_batches = max(0, total_training_batches - completed_batches)
                seconds_per_batch = elapsed / completed_batches
                eta_seconds = remaining_batches * seconds_per_batch
                next_log_batch = min(
                    total_batches,
                    ((batch_index // progress_interval) + 1) * progress_interval,
                )
                batches_to_next_log = max(0, next_log_batch - batch_index)
                eta_next_log = batches_to_next_log * seconds_per_batch
                if batch_index == total_batches:
                    eta_next_log = 0.0
                progress = _progress_bar(completed_batches, total_training_batches)
                percent = 100.0 * completed_batches / total_training_batches
                print(
                    f"  {progress} {percent:5.1f}% "
                    f"batch {batch_index}/{total_batches}: loss={total / batches:.4f} "
                    f"tokens={total_tokens} tok/s={tokens_per_second:.1f} "
                    f"ETA={eta_seconds:.0f}s ETA to next log={eta_next_log:.0f}s",
                    flush=True,
                )
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
    elapsed = max(time.perf_counter() - started_at, 1e-9)
    save_checkpoint(
        args.checkpoint,
        model,
        optimizer,
        step=total_steps,
        epoch=final_epoch,
        loss=final_loss,
        metadata={
            "model": {
                "max_seq_len": args.sequence_length,
                "d_model": d_model,
                "n_heads": n_heads,
                "n_layers": n_layers,
                "model_size": args.model_size,
                "parameter_count": model.parameter_count(),
            },
            "tokenizer": {
                "type": args.tokenizer,
                "vocabulary": list(tokenizer.vocabulary),
                **({"merges": [list(merge) for merge in tokenizer.merges]} if args.tokenizer == "bpe" else {}),
            },
            "training": {
                "train_losses": losses,
                "validation_losses": validation_losses,
                "validation_split": config.validation_split,
                "patience": config.patience,
                "best_validation_loss": best_validation_loss if validation_loader is not None else None,
                "best_epoch": best_epoch if validation_loader is not None else None,
                "seed": args.seed,
                "total_steps": total_steps,
                "total_tokens": total_tokens,
                "tokens_per_second": total_tokens / elapsed,
            },
        },
    )
    print(f"trained {total_tokens} tokens in {elapsed:.1f}s ({total_tokens / elapsed:.1f} tok/s)")
    print(f"saved checkpoint to {args.checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
