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
    assert metadata["training"]["validation_examples"] == 2
    assert metadata["training"]["validation_sampling_ratios"] == {
        "base": 0.5,
        "personality": 0.5,
    }
    assert metadata["training"]["validation_loss_unit"] == "per_assistant_token"


def test_sft_cli_balances_emotion_personality_and_openassistant_data(tmp_path):
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

    def write_dataset(path, prefix):
        records = [
            {
                "state": DEFAULT_STATE,
                "messages": [
                    {"role": "user", "content": f"{prefix} question {index}"},
                    {"role": "assistant", "content": f"{prefix} response {index}"},
                ],
            }
            for index in range(2)
        ]
        path.write_text(
            "\n".join(json.dumps(record) for record in records) + "\n",
            encoding="utf-8",
        )

    base_data = tmp_path / "base.jsonl"
    personality_data = tmp_path / "personality.jsonl"
    emotion_data = tmp_path / "emotion.jsonl"
    smol_data = tmp_path / "smol.jsonl"
    blended_data = tmp_path / "blended.jsonl"
    ultrachat_data = tmp_path / "ultrachat.jsonl"
    gsm8k_data = tmp_path / "gsm8k.jsonl"
    write_dataset(base_data, "OpenAssistant")
    write_dataset(personality_data, "CatAI")
    write_dataset(emotion_data, "Emotion")
    write_dataset(smol_data, "Smol-SmolTalk")
    write_dataset(blended_data, "Blended Skill Talk")
    write_dataset(ultrachat_data, "UltraChat")
    write_dataset(gsm8k_data, "GSM8K")

    output = tmp_path / "balanced-sft.pt"
    result = main([
        "--base-checkpoint", str(base_checkpoint),
        "--dataset", str(base_data),
        "--personality-dataset", str(personality_data),
        "--personality-ratio", "0.05",
        "--emotion-dataset", str(emotion_data),
        "--emotion-ratio", "0.1",
        "--smol-smoltalk-dataset", str(smol_data),
        "--smol-smoltalk-ratio", "0.1",
        "--blended-skill-talk-dataset", str(blended_data),
        "--blended-skill-talk-ratio", "0.05",
        "--ultrachat-dataset", str(ultrachat_data),
        "--ultrachat-ratio", "0.15",
        "--gsm8k-dataset", str(gsm8k_data),
        "--gsm8k-ratio", "0.05",
        "--samples-per-epoch", "4",
        "--checkpoint", str(output),
        "--epochs", "1",
        "--batch-size", "1",
        "--learning-rate", "0.001",
        "--sequence-length", "1024",
        "--device", "cpu",
    ])

    assert result == 0
    metadata = load_checkpoint_metadata(output)
    training = metadata["training"]
    assert training["base_examples"] == 2
    assert training["personality_examples"] == 2
    assert training["emotion_examples"] == 2
    assert training["sampling_ratios"] == {
        "base": 0.5,
        "emotion": 0.1,
        "personality": 0.05,
        "smol_smoltalk": 0.1,
        "blended_skill_talk": 0.05,
        "ultrachat": 0.15,
        "gsm8k": 0.05,
    }
    assert training["emotion_ratio"] == 0.1
    assert training["smol_smoltalk_examples"] == 2
    assert training["smol_smoltalk_ratio"] == 0.1
    assert training["blended_skill_talk_examples"] == 2
    assert training["blended_skill_talk_ratio"] == 0.05
    assert training["ultrachat_examples"] == 2
    assert training["ultrachat_training_examples"] == 2
    assert training["ultrachat_ratio"] == 0.15
    assert training["gsm8k_examples"] == 2
    assert training["gsm8k_training_examples"] == 2
    assert training["gsm8k_ratio"] == 0.05
    assert training["samples_per_epoch"] == 4
    assert training["total_steps"] == 4


def test_sft_cli_rejects_a_base_checkpoint_below_the_required_training_ratio(tmp_path):
    tokenizer = CharTokenizer.default()
    model = CatAI(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=1024,
        d_model=16,
        n_heads=4,
        n_layers=1,
    )
    base_checkpoint = tmp_path / "undertrained.pt"
    save_checkpoint(
        base_checkpoint,
        model,
        torch.optim.AdamW(model.parameters(), lr=1e-3),
        step=10,
        metadata={
            "model": {
                "max_seq_len": 1024,
                "d_model": 16,
                "n_heads": 4,
                "n_layers": 1,
                "parameter_count": 1000,
            },
            "tokenizer": {
                "type": "char",
                "vocabulary": list(tokenizer.vocabulary),
            },
            "training": {"total_tokens": 1000, "document_eos_version": 1},
        },
    )

    import pytest

    with pytest.raises(ValueError, match="base checkpoint is undertrained"):
        main([
            "--base-checkpoint", str(base_checkpoint),
            "--dataset", str(tmp_path / "not-needed-before-guard.jsonl"),
            "--minimum-tokens-per-parameter", "20",
        ])


def test_sft_cli_rejects_a_checkpoint_without_document_boundaries(tmp_path):
    tokenizer = CharTokenizer.default()
    model = CatAI(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=1024,
        d_model=16,
        n_heads=4,
        n_layers=1,
    )
    checkpoint = tmp_path / "legacy-no-document-eos.pt"
    save_checkpoint(
        checkpoint,
        model,
        torch.optim.AdamW(model.parameters(), lr=1e-3),
        step=100,
        metadata={
            "model": {
                "max_seq_len": 1024,
                "d_model": 16,
                "n_heads": 4,
                "n_layers": 1,
                "parameter_count": 1000,
            },
            "tokenizer": {
                "type": "char",
                "vocabulary": list(tokenizer.vocabulary),
            },
            "training": {"total_tokens": 100000, "document_eos_version": 0},
        },
    )

    import pytest

    with pytest.raises(ValueError, match="trained without explicit per-document EOS"):
        main([
            "--base-checkpoint", str(checkpoint),
            "--dataset", str(tmp_path / "not-needed-before-guard.jsonl"),
            "--minimum-tokens-per-parameter", "20",
        ])
