"""Command-line interface for CatAI supervised fine-tuning."""

from __future__ import annotations

import argparse
import time

import torch
from torch.utils.data import DataLoader, random_split

from .chat_dataset import load_chat_dataset
from .checkpoint import load_checkpoint, load_checkpoint_metadata, save_checkpoint
from .device import resolve_device
from .model import CatAI
from .sft import ChatSupervisedDataset, evaluate_sft, sft_train_step
from .tokenizer import BPETokenizer, CharTokenizer


def build_parser() -> argparse.ArgumentParser:
    """Build the supervised fine-tuning argument parser."""
    parser = argparse.ArgumentParser(description="Fine-tune a CatAI checkpoint on chat data.")
    parser.add_argument("--base-checkpoint", required=True, help="Pretrained CatAI checkpoint.")
    parser.add_argument("--dataset", required=True, help="JSONL instruction/chat dataset.")
    parser.add_argument("--checkpoint", default="catai-sft.pt", help="Fine-tuned checkpoint output path.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--validation-split", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    return parser


def _load_tokenizer(metadata: dict) -> CharTokenizer | BPETokenizer:
    tokenizer_type = metadata.get("type")
    vocabulary = metadata.get("vocabulary")
    if not isinstance(vocabulary, list):
        raise ValueError("checkpoint has invalid tokenizer vocabulary")

    if tokenizer_type == "char":
        return CharTokenizer(tuple(vocabulary))

    if tokenizer_type == "bpe":
        merges = metadata.get("merges", [])
        if not isinstance(merges, list):
            raise ValueError("checkpoint has invalid BPE merge metadata")
        try:
            normalized_merges = tuple(tuple(pair) for pair in merges)
        except TypeError as exc:
            raise ValueError("checkpoint has invalid BPE merge metadata") from exc
        return BPETokenizer(tuple(vocabulary), normalized_merges)

    raise ValueError("checkpoint has unsupported tokenizer metadata")


def main(argv: list[str] | None = None) -> int:
    """Fine-tune a pretrained CatAI model on supervised chat examples."""
    args = build_parser().parse_args(argv)
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if args.grad_clip <= 0:
        raise ValueError("--grad-clip must be positive")
    if not 0.0 <= args.validation_split < 1.0:
        raise ValueError("--validation-split must be in [0, 1)")
    if args.patience is not None and args.patience < 1:
        raise ValueError("--patience must be positive")
    if args.sequence_length is not None and args.sequence_length < 1:
        raise ValueError("--sequence-length must be positive")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")

    metadata = load_checkpoint_metadata(args.base_checkpoint)
    model_metadata = metadata.get("model")
    tokenizer_metadata = metadata.get("tokenizer")
    if not isinstance(model_metadata, dict) or not isinstance(tokenizer_metadata, dict):
        raise ValueError("base checkpoint is missing model/tokenizer metadata")

    tokenizer = _load_tokenizer(tokenizer_metadata)
    sequence_length = (
        args.sequence_length
        if args.sequence_length is not None
        else int(model_metadata["max_seq_len"])
    )

    model = CatAI(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=sequence_length,
        d_model=int(model_metadata["d_model"]),
        n_heads=int(model_metadata["n_heads"]),
        n_layers=int(model_metadata["n_layers"]),
    )
    device = resolve_device(args.device)
    model.to(device)
    base_step = load_checkpoint(args.base_checkpoint, model, map_location=device)

    examples = load_chat_dataset(args.dataset)
    dataset = ChatSupervisedDataset(examples, tokenizer, sequence_length)
    validation_loader = None

    if args.validation_split > 0.0:
        validation_size = max(1, int(len(dataset) * args.validation_split))
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

    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    if validation_dataset is not None:
        validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    losses: list[float] = []
    validation_losses: list[float] = []
    best_validation_loss = float("inf")
    stale_epochs = 0
    total_steps = 0
    started_at = time.perf_counter()

    for epoch in range(args.epochs):
        total_loss = 0.0
        batches = 0
        print(f"epoch {epoch + 1}/{args.epochs}", flush=True)
        for batch in loader:
            batch_input_ids = batch["input_ids"].to(device)
            batch_labels = batch["labels"].to(device)
            total_loss += sft_train_step(
                model,
                optimizer,
                batch_input_ids,
                batch_labels,
                grad_clip=args.grad_clip,
            )
            batches += 1
            total_steps += 1

        if batches == 0:
            raise ValueError("training dataset produced no batches")
        epoch_loss = total_loss / batches
        losses.append(epoch_loss)
        print(f"epoch {epoch + 1}/{args.epochs}: train_loss={epoch_loss:.4f}", flush=True)

        if validation_loader is not None:
            validation_loss = evaluate_sft(model, validation_loader, device=device)
            validation_losses.append(validation_loss)
            print(f"epoch {epoch + 1}/{args.epochs}: val_loss={validation_loss:.4f}", flush=True)
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                stale_epochs = 0
            else:
                stale_epochs += 1
                if args.patience is not None and stale_epochs >= args.patience:
                    print(f"early stopping at epoch {epoch + 1}", flush=True)
                    break

    elapsed = max(time.perf_counter() - started_at, 1e-9)
    save_checkpoint(
        args.checkpoint,
        model,
        optimizer,
        step=base_step + total_steps,
        epoch=len(losses),
        loss=losses[-1],
        metadata={
            "stage": "sft",
            "base_checkpoint": str(args.base_checkpoint),
            "model": dict(model_metadata),
            "tokenizer": dict(tokenizer_metadata),
            "training": {
                "train_losses": losses,
                "validation_losses": validation_losses,
                "validation_split": args.validation_split,
                "best_validation_loss": best_validation_loss if validation_loader is not None else None,
                "patience": args.patience,
                "seed": args.seed,
                "total_steps": total_steps,
                "examples": len(dataset),
                "tokens_per_example": sequence_length,
                "seconds": elapsed,
            },
        },
    )
    print(f"fine-tuned {len(dataset)} chat examples in {elapsed:.1f}s")
    print(f"saved checkpoint to {args.checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
