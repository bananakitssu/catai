from __future__ import annotations

import json

import torch

from catai.checkpoint import load_checkpoint_metadata, save_checkpoint
from catai.model import CatAI
from catai.sft_cli import main
from catai.tokenizer import CharTokenizer


def test_sft_cli_fine_tunes_a_checkpoint(tmp_path):
    tokenizer = CharTokenizer.default()
    model = CatAI(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=32,
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
                "max_seq_len": 32,
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
                "messages": [
                    {"role": "system", "content": "You are CatAI. :3"},
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hiii! :3"},
                ]
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
            "--checkpoint", str(output),
            "--epochs", "1",
            "--batch-size", "1",
            "--learning-rate", "0.001",
            "--sequence-length", "32",
            "--device", "cpu",
        ]
    )

    assert exit_code == 0
    assert output.exists()
    metadata = load_checkpoint_metadata(output)
    assert metadata["stage"] == "sft"
    assert metadata["training"]["examples"] == 1
    assert metadata["training"]["total_steps"] == 1
