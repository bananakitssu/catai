"""Command-line interface for CatAI supervised fine-tuning."""

from __future__ import annotations

import argparse
import copy
import hashlib
import time

import torch
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    WeightedRandomSampler,
    random_split,
)

from .chat_dataset import load_chat_dataset
from .checkpoint import (load_checkpoint, load_checkpoint_metadata, load_checkpoint_state, save_checkpoint)
from .device import resolve_device
from .model import CatAI
from .sft import ChatSupervisedDataset, evaluate_sft, sft_train_step
from .tokenizer import BPETokenizer, CharTokenizer


def build_parser() -> argparse.ArgumentParser:
    """Build the supervised fine-tuning argument parser."""
    parser = argparse.ArgumentParser(description="Fine-tune a CatAI checkpoint on chat data.")
    parser.add_argument("--base-checkpoint", required=True, help="Pretrained CatAI checkpoint.")
    parser.add_argument("--dataset", required=True, help="Base chat dataset JSONL, such as OASST1.")
    parser.add_argument(
        "--personality-dataset",
        default=None,
        help="Optional curated CatAI dataset to upsample separately from the base chat data.",
    )
    parser.add_argument(
        "--personality-ratio",
        type=float,
        default=0.25,
        help="Target share of training samples drawn from --personality-dataset (0 < ratio < 1).",
    )
    parser.add_argument("--checkpoint", default="catai-sft.pt", help="Fine-tuned checkpoint output path.")
    parser.add_argument("--resume-from", default=None, help="Resume model, optimizer, and data position from a recovery checkpoint.")
    parser.add_argument("--recovery-checkpoint", default=None, help="Separate checkpoint updated during training.")
    parser.add_argument("--checkpoint-every-batches", type=int, default=50, help="Save recovery state every N batches.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--validation-split", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=1024)
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
    if args.checkpoint_every_batches < 1:
        raise ValueError("--checkpoint-every-batches must be positive")
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
    if not 0.0 < args.personality_ratio < 1.0:
        raise ValueError("--personality-ratio must be greater than 0 and less than 1")

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

    base_examples = load_chat_dataset(args.dataset)
    base_dataset = ChatSupervisedDataset(
        base_examples, tokenizer, sequence_length
    )
    print(
        f"base chat data: {len(base_examples)} loaded, "
        f"{base_dataset.compacted_examples} compacted to the latest turn, "
        f"{base_dataset.skipped_overlength_examples} overlength examples skipped",
        flush=True,
    )
    personality_dataset = None
    personality_examples_count = 0

    if args.personality_dataset:
        personality_examples = load_chat_dataset(args.personality_dataset)
        personality_examples_count = len(personality_examples)
        personality_dataset = ChatSupervisedDataset(
            personality_examples, tokenizer, sequence_length
        )
        print(
            f"CatAI instruction data: {personality_examples_count} loaded, "
            f"{personality_dataset.compacted_examples} compacted to the latest turn, "
            f"{personality_dataset.skipped_overlength_examples} overlength examples skipped",
            flush=True,
        )
        dataset = ConcatDataset([base_dataset, personality_dataset])
    else:
        dataset = base_dataset

    validation_loader = None
    if personality_dataset is not None:
        base_validation_size = (
            max(1, int(len(base_dataset) * args.validation_split))
            if args.validation_split > 0.0
            else 0
        )
        personality_validation_size = (
            max(1, int(len(personality_dataset) * args.validation_split))
            if args.validation_split > 0.0
            else 0
        )
        if base_validation_size >= len(base_dataset):
            raise ValueError("validation_split leaves no base training samples")
        if personality_validation_size >= len(personality_dataset):
            raise ValueError("validation_split leaves no personality training samples")

        if args.validation_split > 0.0:
            base_train, base_validation = random_split(
                base_dataset,
                [len(base_dataset) - base_validation_size, base_validation_size],
                generator=torch.Generator().manual_seed(args.seed),
            )
            personality_train, personality_validation = random_split(
                personality_dataset,
                [
                    len(personality_dataset) - personality_validation_size,
                    personality_validation_size,
                ],
                generator=torch.Generator().manual_seed(args.seed + 1),
            )
            validation_dataset = ConcatDataset(
                [base_validation, personality_validation]
            )
            validation_loader = DataLoader(
                validation_dataset, batch_size=args.batch_size, shuffle=False
            )
        else:
            base_train = base_dataset
            personality_train = personality_dataset
            validation_dataset = None

        train_dataset = ConcatDataset([base_train, personality_train])
        base_weight = (1.0 - args.personality_ratio) / len(base_train)
        personality_weight = args.personality_ratio / len(personality_train)
        weights = (
            [base_weight] * len(base_train)
            + [personality_weight] * len(personality_train)
        )
        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(train_dataset),
            replacement=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
        loader = DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=sampler
        )
    else:
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
            validation_loader = DataLoader(
                validation_dataset, batch_size=args.batch_size, shuffle=False
            )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    losses: list[float] = []
    validation_losses: list[float] = []
    best_validation_loss = float("inf")
    best_epoch = 0
    best_model_state = None
    best_optimizer_state = None
    stale_epochs = 0
    total_steps = 0
    start_epoch = 0
    resume_batch = 0
    saved_epoch_indices = None
    saved_next_batch_hash = None
    if args.resume_from:
        resume_state = load_checkpoint_state(args.resume_from)
        base_step = load_checkpoint(args.resume_from, model, optimizer, map_location=device)
        start_epoch = int(resume_state.get("epoch_index", 0))
        resume_batch = int(resume_state.get("batch_index", 0))
        saved_epoch_indices = resume_state.get("epoch_indices")
        saved_next_batch_hash = resume_state.get("next_batch_hash")
        losses = list(resume_state.get("losses", []))
        validation_losses = list(resume_state.get("validation_losses", []))
        best_validation_loss = float(resume_state.get("best_validation_loss", float("inf")))
        best_epoch = int(resume_state.get("best_epoch", 0))
        best_model_state = resume_state.get("best_model_state")
        best_optimizer_state = resume_state.get("best_optimizer_state")
        stale_epochs = int(resume_state.get("stale_epochs", 0))
        total_steps = int(resume_state.get("total_steps", 0))
        rng_state = resume_state.get("torch_rng_state")
        if rng_state is not None:
            torch.set_rng_state(rng_state)
        print(
            f"Resuming at epoch {start_epoch + 1}, last completed batch {resume_batch}; "
            f"optimizer step {total_steps}.",
            flush=True,
        )
    else:
        base_step = load_checkpoint(args.base_checkpoint, model, optimizer, map_location=device)
    total_batches = (len(train_dataset) + args.batch_size - 1) // args.batch_size
    progress_interval = max(1, total_batches // 10)
    started_at = time.perf_counter()

    def batch_hash(indices):
        if not indices:
            return None
        sample = train_dataset[indices[0]]["input_ids"][:10].detach().cpu().contiguous()
        return hashlib.sha256(sample.numpy().tobytes()).hexdigest()

    def save_recovery(epoch_index, batch_index, epoch_indices):
        if not args.recovery_checkpoint:
            return
        next_indices = (
            epoch_indices[batch_index:batch_index + args.batch_size]
            if epoch_index < args.epochs and batch_index < len(epoch_indices)
            else []
        )
        next_hash = batch_hash(next_indices)
        extra = {
            "epoch_index": epoch_index,
            "batch_index": batch_index,
            "epoch_indices": epoch_indices,
            "next_batch_hash": next_hash,
            "torch_rng_state": torch.get_rng_state(),
            "losses": losses,
            "validation_losses": validation_losses,
            "best_validation_loss": best_validation_loss,
            "best_epoch": best_epoch,
            "best_model_state": best_model_state,
            "best_optimizer_state": best_optimizer_state,
            "stale_epochs": stale_epochs,
            "total_steps": total_steps,
        }
        save_checkpoint(
            args.recovery_checkpoint, model, optimizer, step=base_step + total_steps,
            epoch=epoch_index, loss=losses[-1] if losses else None,
            metadata={"stage": "sft-recovery", "model": sft_model_metadata_for_resume(),
                      "tokenizer": dict(tokenizer_metadata)},
            extra_state=extra,
        )

    def sft_model_metadata_for_resume():
        result = dict(model_metadata)
        result["max_seq_len"] = sequence_length
        return result

    for epoch in range(start_epoch, args.epochs):
        if saved_epoch_indices is not None and epoch == start_epoch:
            epoch_indices = list(saved_epoch_indices)
        elif personality_dataset is not None:
            generator = torch.Generator().manual_seed(args.seed + epoch)
            epoch_indices = torch.multinomial(
                torch.tensor(weights, dtype=torch.double),
                num_samples=len(train_dataset), replacement=True, generator=generator
            ).tolist()
        else:
            generator = torch.Generator().manual_seed(args.seed + epoch)
            epoch_indices = torch.randperm(len(train_dataset), generator=generator).tolist()
        batches_for_epoch = [
            epoch_indices[i:i + args.batch_size]
            for i in range(0, len(epoch_indices), args.batch_size)
        ]
        if not batches_for_epoch:
            raise ValueError("training dataset produced no batches")
        total_loss = 0.0
        batches = 0
        first_batch = resume_batch if epoch == start_epoch else 0
        print(f"epoch {epoch + 1}/{args.epochs}", flush=True)
        if first_batch and saved_next_batch_hash:
            actual_hash = batch_hash(batches_for_epoch[first_batch])
            print(
                f"resume sanity: last batch index {first_batch}; optimizer step {total_steps}; "
                f"next-batch token hash {actual_hash}",
                flush=True,
            )
            if actual_hash != saved_next_batch_hash:
                raise ValueError("resume sanity check failed: next-batch token hash differs")
        elif epoch == start_epoch and args.resume_from:
            print(
                f"resume sanity: last batch index {first_batch}; optimizer step {total_steps}; "
                f"next-batch token hash {batch_hash(batches_for_epoch[first_batch]) if first_batch < len(batches_for_epoch) else 'epoch-boundary'}",
                flush=True,
            )
        for batch_offset in range(first_batch, len(batches_for_epoch)):
            indices = batches_for_epoch[batch_offset]
            batch = {
                "input_ids": torch.stack([train_dataset[i]["input_ids"] for i in indices]),
                "labels": torch.stack([train_dataset[i]["labels"] for i in indices]),
            }
            batch_input_ids = batch["input_ids"].to(device)
            batch_labels = batch["labels"].to(device)
            total_loss += sft_train_step(
                model, optimizer, batch_input_ids, batch_labels, grad_clip=args.grad_clip
            )
            batches += 1
            total_steps += 1
            completed_batch = batch_offset + 1
            if completed_batch % args.checkpoint_every_batches == 0:
                save_recovery(epoch, completed_batch, epoch_indices)
            if completed_batch % progress_interval == 0 or completed_batch == len(batches_for_epoch):
                elapsed = max(time.perf_counter() - started_at, 1e-9)
                completed_total = epoch * total_batches + completed_batch
                remaining = max(0, args.epochs * total_batches - completed_total)
                eta = remaining * elapsed / max(completed_total - start_epoch * total_batches, 1)
                print(
                    f"  batch {completed_batch}/{len(batches_for_epoch)} "
                    f"loss={total_loss / max(batches, 1):.4f} ETA={eta:.0f}s",
                    flush=True,
                )
        if batches == 0 and first_batch == 0:
            raise ValueError("training dataset produced no batches")
        if batches:
            epoch_loss = total_loss / batches
            if len(losses) < epoch + 1:
                losses.append(epoch_loss)
            else:
                losses[epoch] = epoch_loss
            print(f"epoch {epoch + 1}/{args.epochs}: train_loss={epoch_loss:.4f}", flush=True)
        if validation_loader is not None:
            validation_loss = evaluate_sft(model, validation_loader, device=device)
            if len(validation_losses) < epoch + 1:
                validation_losses.append(validation_loss)
            else:
                validation_losses[epoch] = validation_loss
            print(f"epoch {epoch + 1}/{args.epochs}: val_loss={validation_loss:.4f}", flush=True)
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_epoch = epoch + 1
                best_model_state = copy.deepcopy(model.state_dict())
                best_optimizer_state = copy.deepcopy(optimizer.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
                if args.patience is not None and stale_epochs >= args.patience:
                    print(f"early stopping at epoch {epoch + 1}", flush=True)
                    save_recovery(epoch + 1, 0, [])
                    break
        resume_batch = 0
        saved_epoch_indices = None
        save_recovery(epoch + 1, 0, [])


    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        if best_optimizer_state is not None:
            optimizer.load_state_dict(best_optimizer_state)
        print(
            f"restored best validation checkpoint from epoch {best_epoch}",
            flush=True,
        )

    elapsed = max(time.perf_counter() - started_at, 1e-9)
    selected_epoch = best_epoch if best_epoch else len(losses)
    selected_loss = losses[selected_epoch - 1]
    sft_model_metadata = dict(model_metadata)
    sft_model_metadata["max_seq_len"] = sequence_length

    save_checkpoint(
        args.checkpoint,
        model,
        optimizer,
        step=base_step + total_steps,
        epoch=len(losses),
        loss=selected_loss,
        metadata={
            "stage": "sft",
            "base_checkpoint": str(args.base_checkpoint),
            "model": sft_model_metadata,
            "tokenizer": dict(tokenizer_metadata),
            "training": {
                "train_losses": losses,
                "validation_losses": validation_losses,
                "validation_split": args.validation_split,
                "best_validation_loss": best_validation_loss if validation_loader is not None else None,
                "best_epoch": best_epoch if best_epoch else None,
                "selected_epoch": selected_epoch,
                "patience": args.patience,
                "seed": args.seed,
                "total_steps": total_steps,
                "examples": len(dataset),
                "base_examples": len(base_examples),
                "base_training_examples": len(base_dataset),
                "base_compacted_examples": base_dataset.compacted_examples,
                "base_skipped_overlength_examples": base_dataset.skipped_overlength_examples,
                "personality_examples": personality_examples_count,
                "personality_training_examples": (
                    len(personality_dataset) if personality_dataset is not None else 0
                ),
                "personality_compacted_examples": (
                    personality_dataset.compacted_examples
                    if personality_dataset is not None else 0
                ),
                "personality_skipped_overlength_examples": (
                    personality_dataset.skipped_overlength_examples
                    if personality_dataset is not None else 0
                ),
                "personality_ratio": (
                    args.personality_ratio
                    if personality_dataset is not None
                    else None
                ),
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
