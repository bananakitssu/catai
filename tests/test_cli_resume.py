from pathlib import Path

from catai.checkpoint import load_checkpoint_metadata, load_checkpoint_state
from catai.cli import main


def test_base_training_can_resume_from_a_mid_epoch_checkpoint(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("a small cat learns to talk. " * 100, encoding="utf-8")
    recovery = tmp_path / "base-recovery.pt"
    checkpoint = tmp_path / "base.pt"

    common_args = [
        "--corpus", str(corpus),
        "--checkpoint", str(checkpoint),
        "--epochs", "1",
        "--batch-size", "2",
        "--sequence-length", "8",
        "--model-size", "tiny",
        "--d-model", "8",
        "--heads", "2",
        "--layers", "1",
        "--tokenizer", "char",
        "--validation-split", "0.1",
        "--seed", "42",
        "--device", "cpu",
    ]

    assert main(
        common_args
        + [
            "--recovery-checkpoint", str(recovery),
            "--checkpoint-every-batches", "1",
            "--max-batches-per-run-fraction", "0.5",
        ]
    ) == 0
    assert recovery.exists()
    assert not checkpoint.exists()

    first_state = load_checkpoint_state(recovery)
    first_steps = int(first_state["total_steps"])
    assert first_steps > 0
    assert first_state["stage"] == "base-training-recovery"

    assert main(
        common_args
        + [
            "--resume-from", str(recovery),
            "--recovery-checkpoint", str(recovery),
            "--checkpoint-every-batches", "1",
        ]
    ) == 0
    assert checkpoint.exists()

    metadata = load_checkpoint_metadata(checkpoint)
    assert metadata["training"]["total_steps"] > first_steps
    assert len(metadata["training"]["train_losses"]) == 1
