"""Command-line interface for CatAI training."""

from __future__ import annotations

import argparse
import math
import time

import torch
from torch.utils.data import random_split

from .batching import make_dataloader
from .checkpoint import (
    load_checkpoint,
    load_checkpoint_metadata,
    load_checkpoint_state,
    save_checkpoint,
)
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
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=1024,
        help="Training context length in tokens (default: 1024).",
    )
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
    parser.add_argument(
        "--tokenizer-sample-chars",
        type=int,
        default=250_000,
        help="Number of leading corpus characters used to learn BPE merges.",
    )
    parser.add_argument("--max-tokens", type=int, default=None, help="Cap the training token budget.")
    parser.add_argument("--validation-split", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=None, help="Early-stop after this many non-improving validation epochs.")
    parser.add_argument("--resume-from", default=None, help="Resume model, optimizer, and data position from a recovery checkpoint.")
    parser.add_argument("--recovery-checkpoint", default=None, help="Checkpoint updated during training so a later job can resume.")
    parser.add_argument("--checkpoint-every-batches", type=int, default=50, help="Save recovery state every N batches.")
    parser.add_argument("--max-batches-per-run-fraction", type=float, default=None, help="Stop after this fraction of one epoch, saving a recovery checkpoint for the next job.")
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
    """Train CatAI from a text corpus, with resumable checkpoints."""
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
    if args.tokenizer == "bpe" and args.vocab_size < 3:
        raise ValueError("--vocab-size must be at least 3")
    if args.tokenizer_sample_chars < 1:
        raise ValueError("--tokenizer-sample-chars must be positive")
    if args.checkpoint_every_batches < 1:
        raise ValueError("--checkpoint-every-batches must be positive")
    if args.max_batches_per_run_fraction is not None and not (
        0.0 < args.max_batches_per_run_fraction <= 1.0
    ):
        raise ValueError("--max-batches-per-run-fraction must be in (0, 1]")

    model_config = ModelConfig.from_preset(args.model_size)
    if args.model_size != "tiny":
        d_model, n_heads, n_layers = (
            model_config.d_model,
            model_config.n_heads,
            model_config.n_layers,
        )
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
        tokenizer_sample_chars=args.tokenizer_sample_chars,
    )
    window_stride = (
        args.window_stride if args.window_stride is not None else args.sequence_length
    )
    print(
        f"tokenizer={args.tokenizer} vocab_size={tokenizer.vocab_size} "
        f"encoded_tokens={len(tokens):,} sequence_length={args.sequence_length}",
        flush=True,
    )
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

    if validation_dataset is not None:
        validation_loader = make_dataloader(
            validation_dataset, batch_size=args.batch_size, shuffle=False
        )

    device = resolve_device(args.device)
    model = CatAI(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=args.sequence_length,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    model_metadata = {
        "max_seq_len": args.sequence_length,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "model_size": args.model_size,
        "parameter_count": model.parameter_count(),
    }
    tokenizer_metadata = {
        "type": args.tokenizer,
        "vocabulary": list(tokenizer.vocabulary),
        **(
            {
                "merges": [list(merge) for merge in tokenizer.merges],
                "training_sample_char_limit": args.tokenizer_sample_chars,
            }
            if args.tokenizer == "bpe"
            else {}
        ),
    }
    resume_config = {
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "grad_clip": config.grad_clip,
        "sequence_length": args.sequence_length,
        "window_stride": window_stride,
        "model_size": args.model_size,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "tokenizer_type": args.tokenizer,
        "vocab_size": tokenizer.vocab_size,
        "tokenizer_sample_chars": args.tokenizer_sample_chars,
        "max_tokens": config.max_tokens,
        "validation_split": config.validation_split,
        "seed": args.seed,
        "training_samples": len(train_dataset),
        "validation_samples": len(validation_dataset) if validation_dataset is not None else 0,
    }

    losses: list[float] = []
    validation_losses: list[float] = []
    best_validation_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    base_step = 0
    total_steps = 0
    total_tokens = 0
    start_epoch = 0
    resume_batch = 0
    saved_epoch_indices = None
    resume_epoch_loss = 0.0
    resume_epoch_batches = 0

    if args.resume_from:
        saved_metadata = load_checkpoint_metadata(args.resume_from)
        if saved_metadata.get("resume_config") != resume_config:
            raise ValueError(
                "resume sanity check failed: base-training configuration or dataset size changed"
            )
        resume_state = load_checkpoint_state(args.resume_from)
        if resume_state.get("stage") != "base-training-recovery":
            raise ValueError("resume checkpoint is not a CatAI base-training recovery checkpoint")
        restored_step = load_checkpoint(
            args.resume_from, model, optimizer, map_location=device
        )
        base_step = int(resume_state.get("base_step", 0))
        total_steps = int(resume_state.get("total_steps", 0))
        if restored_step != base_step + total_steps:
            raise ValueError(
                f"resume sanity check failed: checkpoint step {restored_step} "
                f"does not match saved training step {base_step + total_steps}"
            )
        start_epoch = int(resume_state.get("epoch_index", 0))
        resume_batch = int(resume_state.get("batch_index", 0))
        saved_epoch_indices = resume_state.get("epoch_indices")
        losses = list(resume_state.get("losses", []))
        validation_losses = list(resume_state.get("validation_losses", []))
        best_validation_loss = float(
            resume_state.get("best_validation_loss", float("inf"))
        )
        best_epoch = int(resume_state.get("best_epoch", 0))
        stale_epochs = int(resume_state.get("stale_epochs", 0))
        total_tokens = int(resume_state.get("total_tokens", 0))
        resume_epoch_loss = float(resume_state.get("current_epoch_loss", 0.0))
        resume_epoch_batches = int(resume_state.get("current_epoch_batches", 0))
        rng_state = resume_state.get("torch_rng_state")
        if rng_state is not None:
            torch.set_rng_state(rng_state)
        if start_epoch > config.epochs:
            raise ValueError("resume checkpoint is already beyond the requested epoch count")
        print(
            f"Resuming at epoch {start_epoch + 1}, last completed batch {resume_batch}; "
            f"optimizer step {total_steps}.",
            flush=True,
        )

    total_batches = (len(train_dataset) + args.batch_size - 1) // args.batch_size
    if total_batches < 1:
        raise ValueError("training dataset produced no batches")
    max_batches_this_run = None
    if args.max_batches_per_run_fraction is not None:
        max_batches_this_run = max(
            1, math.ceil(total_batches * args.max_batches_per_run_fraction)
        )
        print(
            f"This job will process at most {max_batches_this_run:,} batches "
            f"({args.max_batches_per_run_fraction:.0%} of one epoch).",
            flush=True,
        )

    progress_interval = max(1, total_batches // 10)
    started_at = time.perf_counter()
    batches_this_run = 0
    paused = False

    def save_recovery(
        epoch_index: int,
        batch_index: int,
        epoch_indices: list[int] | None,
        current_epoch_loss: float,
        current_epoch_batches: int,
    ) -> None:
        if not args.recovery_checkpoint:
            return
        extra_state = {
            "stage": "base-training-recovery",
            "epoch_index": epoch_index,
            "batch_index": batch_index,
            "epoch_indices": epoch_indices,
            "torch_rng_state": torch.get_rng_state(),
            "losses": losses,
            "validation_losses": validation_losses,
            "best_validation_loss": best_validation_loss,
            "best_epoch": best_epoch,
            "stale_epochs": stale_epochs,
            "total_steps": total_steps,
            "total_tokens": total_tokens,
            "base_step": base_step,
            "current_epoch_loss": current_epoch_loss,
            "current_epoch_batches": current_epoch_batches,
        }
        save_checkpoint(
            args.recovery_checkpoint,
            model,
            optimizer,
            step=base_step + total_steps,
            epoch=epoch_index,
            loss=losses[-1] if losses else None,
            metadata={
                "model": model_metadata,
                "tokenizer": tokenizer_metadata,
                "resume_config": resume_config,
            },
            extra_state=extra_state,
        )

    for epoch in range(start_epoch, config.epochs):
        if saved_epoch_indices is not None and epoch == start_epoch:
            epoch_indices = list(saved_epoch_indices)
        else:
            generator = torch.Generator().manual_seed(args.seed + epoch)
            epoch_indices = torch.randperm(
                len(train_dataset), generator=generator
            ).tolist()
        if len(epoch_indices) != len(train_dataset):
            raise ValueError("resume sanity check failed: saved epoch ordering has changed")

        batches_for_epoch = [
            epoch_indices[index:index + args.batch_size]
            for index in range(0, len(epoch_indices), args.batch_size)
        ]
        if not batches_for_epoch:
            raise ValueError("training dataset produced no batches")
        if epoch == start_epoch:
            first_batch = resume_batch
            epoch_loss = resume_epoch_loss
            epoch_batches = resume_epoch_batches
        else:
            first_batch = 0
            epoch_loss = 0.0
            epoch_batches = 0
        if first_batch < 0 or first_batch > len(batches_for_epoch):
            raise ValueError("resume sanity check failed: saved batch position is invalid")

        print(f"epoch {epoch + 1}/{config.epochs}", flush=True)
        for batch_offset in range(first_batch, len(batches_for_epoch)):
            indices = batches_for_epoch[batch_offset]
            batch = torch.stack([train_dataset[index] for index in indices]).to(device)
            epoch_loss += train_step(
                model, optimizer, batch, grad_clip=config.grad_clip
            )
            epoch_batches += 1
            total_steps += 1
            total_tokens += batch.numel() - batch.size(0)
            batches_this_run += 1
            completed_batch = batch_offset + 1

            if completed_batch % args.checkpoint_every_batches == 0:
                save_recovery(
                    epoch,
                    completed_batch,
                    epoch_indices,
                    epoch_loss,
                    epoch_batches,
                )

            if completed_batch % progress_interval == 0 or completed_batch == total_batches:
                elapsed = max(time.perf_counter() - started_at, 1e-9)
                tokens_per_second = total_tokens / elapsed
                completed_total = epoch * total_batches + completed_batch
                total_training_batches = config.epochs * total_batches
                remaining_batches = max(0, total_training_batches - completed_total)
                eta_seconds = remaining_batches * elapsed / max(batches_this_run, 1)
                progress = _progress_bar(completed_total, total_training_batches)
                percent = 100.0 * completed_total / total_training_batches
                print(
                    f"  {progress} {percent:5.1f}% "
                    f"batch {completed_batch}/{total_batches}: "
                    f"loss={epoch_loss / epoch_batches:.4f} "
                    f"tokens={total_tokens} tok/s={tokens_per_second:.1f} "
                    f"ETA={eta_seconds:.0f}s",
                    flush=True,
                )

            if (
                max_batches_this_run is not None
                and batches_this_run >= max_batches_this_run
                and completed_batch < len(batches_for_epoch)
            ):
                save_recovery(
                    epoch,
                    completed_batch,
                    epoch_indices,
                    epoch_loss,
                    epoch_batches,
                )
                paused = True
                break

        if paused:
            break
        if epoch_batches == 0:
            raise ValueError("training dataset produced no batches")

        epoch_loss_value = epoch_loss / epoch_batches
        if len(losses) < epoch + 1:
            losses.append(epoch_loss_value)
        else:
            losses[epoch] = epoch_loss_value
        print(
            f"epoch {epoch + 1}/{config.epochs}: train_loss={epoch_loss_value:.4f}",
            flush=True,
        )

        should_stop = False
        if validation_loader is not None:
            val_loss = evaluate(model, validation_loader, device=device)
            if len(validation_losses) < epoch + 1:
                validation_losses.append(val_loss)
            else:
                validation_losses[epoch] = val_loss
            print(
                f"epoch {epoch + 1}/{config.epochs}: val_loss={val_loss:.4f}",
                flush=True,
            )
            if val_loss < best_validation_loss:
                best_validation_loss = val_loss
                best_epoch = epoch + 1
                stale_epochs = 0
                print("  new best validation loss", flush=True)
            else:
                stale_epochs += 1
                if config.patience is not None and stale_epochs >= config.patience:
                    print(f"early stopping at epoch {epoch + 1}", flush=True)
                    should_stop = True

        save_recovery(epoch + 1, 0, None, 0.0, 0)
        saved_epoch_indices = None
        resume_batch = 0
        resume_epoch_loss = 0.0
        resume_epoch_batches = 0

        if should_stop:
            break
        if (
            max_batches_this_run is not None
            and batches_this_run >= max_batches_this_run
            and epoch + 1 < config.epochs
        ):
            paused = True
            break

    if paused:
        print(
            f"Training paused after {batches_this_run:,} batches; "
            f"recovery checkpoint saved to {args.recovery_checkpoint}.",
            flush=True,
        )
        return 0

    if not losses:
        raise ValueError("training completed without producing an epoch loss")
    final_epoch = len(losses)
    final_loss = losses[-1]
    elapsed = max(time.perf_counter() - started_at, 1e-9)
    save_checkpoint(
        args.checkpoint,
        model,
        optimizer,
        step=base_step + total_steps,
        epoch=final_epoch,
        loss=final_loss,
        metadata={
            "model": model_metadata,
            "tokenizer": tokenizer_metadata,
            "training": {
                "train_losses": losses,
                "validation_losses": validation_losses,
                "validation_split": config.validation_split,
                "patience": config.patience,
                "best_validation_loss": (
                    best_validation_loss if validation_loader is not None else None
                ),
                "best_epoch": best_epoch if validation_loader is not None else None,
                "seed": args.seed,
                "total_steps": total_steps,
                "total_tokens": total_tokens,
                "tokens_per_second": total_tokens / elapsed,
            },
        },
    )
    print(
        f"trained {total_tokens} tokens in {elapsed:.1f}s "
        f"({total_tokens / elapsed:.1f} tok/s)"
    )
    print(f"saved checkpoint to {args.checkpoint}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
