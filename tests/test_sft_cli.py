from __future__ import annotations

import json

import torch

from catai.checkpoint import load_checkpoint, load_checkpoint_metadata, save_checkpoint
from catai.chat_dataset import DEFAULT_STATE
from catai.model import CatAI
import catai.sft_cli as sft_cli
from catai.sft_cli import main
from catai.tokenizer import CharTokenizer


def test_sft_cli_fine_tunes_a_checkpoint(tmp_path):
    tokenizer = CharTokenizer.default()
    model = CatAI(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=1024,
        d_model=16,
        n_heads=4,
        n_layers=1,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    base_checkpoint = tmp_path / "base.pt"
    save_checkpoint(
        base_checkpoint,
        model,
        optimizer,
        step=0,
        metadata={
            "model": {
                "max_seq_len": 1024,
                "d_model": 16,
                "n_heads": 4,
                "n_layers": 1,
            },
            "tokenizer": {
                "type": "char",
                "vocabulary": list(tokenizer.vocabulary),
            },
        },
    )

    dataset = tmp_path / "chat.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "state": DEFAULT_STATE,
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hiii! :3"},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    personality_dataset = tmp_path / "personality.jsonl"
    personality_dataset.write_text(
        json.dumps(
            {
                "state": DEFAULT_STATE,
                "messages": [
                    {"role": "user", "content": "Say hello like a cat."},
                    {"role": "assistant", "content": "*ears perk up* Hello! :3"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output = tmp_path / "sft.pt"
    exit_code = main(
        [
            "--base-checkpoint", str(base_checkpoint),
            "--dataset", str(dataset),
            "--personality-dataset", str(personality_dataset),
            "--personality-ratio", "0.5",
            "--checkpoint", str(output),
            "--epochs", "1",
            "--batch-size", "1",
            "--learning-rate", "0.001",
            "--sequence-length", "1024",
            "--device", "cpu",
        ]
    )

    assert exit_code == 0
    assert output.exists()
    metadata = load_checkpoint_metadata(output)
    assert metadata["stage"] == "sft"
    assert metadata["model"]["max_seq_len"] == 1024
    assert metadata["training"]["examples"] == 2
    assert metadata["training"]["base_examples"] == 1
    assert metadata["training"]["personality_examples"] == 1
    assert metadata["training"]["personality_ratio"] == 0.5
    assert metadata["training"]["total_steps"] == 2



def test_sft_cli_restores_the_best_validation_epoch(tmp_path, monkeypatch):
    tokenizer = CharTokenizer.default()
    model = CatAI(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=1024,
        d_model=16,
        n_heads=4,
        n_layers=1,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    base_checkpoint = tmp_path / "base.pt"
    save_checkpoint(
        base_checkpoint,
        model,
        optimizer,
        step=0,
        metadata={
            "model": {
                "max_seq_len": 1024,
                "d_model": 16,
                "n_heads": 4,
                "n_layers": 1,
            },
            "tokenizer": {
                "type": "char",
                "vocabulary": list(tokenizer.vocabulary),
            },
        },
    )

    def write_dataset(path, prefixes):
        records = []
        for prefix in prefixes:
            records.append({
                "state": DEFAULT_STATE,
                "messages": [
                    {"role": "user", "content": f"{prefix} question"},
                    {"role": "assistant", "content": f"{prefix} answer"},
                ],
            })
        path.write_text(
            "\n".join(json.dumps(record) for record in records) + "\n",
            encoding="utf-8",
        )

    dataset = tmp_path / "chat.jsonl"
    personality_dataset = tmp_path / "personality.jsonl"
    write_dataset(dataset, ["first", "second"])
    write_dataset(personality_dataset, ["third", "fourth"])

    training_step = 0

    def fake_train_step(model, optimizer, input_ids, labels, grad_clip=None):
        nonlocal training_step
        training_step += 1
        with torch.no_grad():
            model.token_embedding.weight.fill_(float(training_step))
        return 1.0

    validation_losses = iter([1.0, 0.5, 0.75])
    monkeypatch.setattr(sft_cli, "sft_train_step", fake_train_step)
    monkeypatch.setattr(
        sft_cli,
        "evaluate_sft",
        lambda *args, **kwargs: next(validation_losses),
    )

    output = tmp_path / "best-sft.pt"
    exit_code = main(
        [
            "--base-checkpoint", str(base_checkpoint),
            "--dataset", str(dataset),
            "--personality-dataset", str(personality_dataset),
            "--personality-ratio", "0.5",
            "--checkpoint", str(output),
            "--epochs", "3",
            "--batch-size", "1",
            "--learning-rate", "0.001",
            "--validation-split", "0.5",
            "--patience", "3",
            "--sequence-length", "1024",
            "--device", "cpu",
        ]
    )

    assert exit_code == 0
    assert training_step == 6

    restored = CatAI(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=1024,
        d_model=16,
        n_heads=4,
        n_layers=1,
    )
    load_checkpoint(output, restored)
    assert torch.all(restored.token_embedding.weight == 4.0)

    metadata = load_checkpoint_metadata(output)
    assert metadata["training"]["best_epoch"] == 2
    assert metadata["training"]["selected_epoch"] == 2
    assert metadata["training"]["best_validation_loss"] == 0.5
