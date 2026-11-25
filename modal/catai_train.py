"""Run CatAI's unchanged base-training or SFT workload on Modal GPUs.

Create the persistent Volume and upload private inputs before invoking this app.
The local entrypoint requires an explicit paid-compute acknowledgement.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import modal

VOLUME_NAME = "catai-training"
MOUNT_PATH = Path("/mnt/catai")
INPUTS = MOUNT_PATH / "inputs"
OUTPUTS = MOUNT_PATH / "outputs"

volume = modal.Volume.from_name(VOLUME_NAME)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.2",
        "numpy>=1.26",
    )
    .add_local_dir("src", remote_path="/root/catai/src")
    .add_local_file("pyproject.toml", remote_path="/root/catai/pyproject.toml")
    .add_local_file("scripts/export_inference.py", remote_path="/root/catai/scripts/export_inference.py")
    .env({"PYTHONPATH": "/root/catai/src"})
)

app = modal.App("catai-training")


def require_files(names: list[str]) -> dict[str, Path]:
    found = {name: INPUTS / name for name in names}
    missing = [str(path) for path in found.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing Modal Volume inputs: " + ", ".join(missing)
            + ". Upload them with modal volume put catai-training LOCAL_PATH inputs/."
        )
    return found


@app.function(
    image=image,
    gpu="L4",
    cpu=8,
    memory=32768,
    timeout=6 * 60 * 60,
    volumes={str(MOUNT_PATH): volume},
)
def run_training(mode: str) -> dict[str, str | bool]:
    """Run one complete SFT job or one resumable base-training segment."""
    if mode not in {"base", "sft"}:
        raise ValueError("mode must be 'base' or 'sft'")

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    if mode == "base":
        files = require_files(["corpus.txt"])
        checkpoint = OUTPUTS / "catai.pt"
        recovery = OUTPUTS / "catai-base-recovery.pt"
        if checkpoint.is_file():
            return {"completed": True, "checkpoint": str(checkpoint), "message": "Base checkpoint already exists."}
        command = [
            sys.executable, "-m", "catai",
            "--corpus", str(files["corpus.txt"]),
            "--checkpoint", str(checkpoint),
            "--recovery-checkpoint", str(recovery),
            "--checkpoint-every-batches", "50",
            "--epochs", "1",
            "--batch-size", "4",
            "--sequence-length", "1024",
            "--model-size", "base",
            "--tokenizer", "bpe",
            "--vocab-size", "2048",
            "--tokenizer-sample-chars", "250000",
            "--max-tokens", "850000000",
            "--validation-split", "0.05",
            "--minimum-tokens-per-parameter", "20",
            "--warmup-fraction", "0.03",
            "--minimum-learning-rate-ratio", "0.1",
            "--seed", "42",
            "--device", "cuda",
            "--precision", "auto",
            "--max-batches-per-run-fraction", "0.15",
        ]
        if recovery.is_file():
            command.extend(["--resume-from", str(recovery)])
    else:
        required = [
            "catai.pt",
            "oasst_chat.jsonl",
            "instruction_chat.jsonl",
            "smol_smoltalk_chat.jsonl",
            "blended_skill_talk_chat.jsonl",
            "ultrachat_200k_chat.jsonl",
            "expanded_sft_chat.jsonl",
            "markdown_sft_chat.jsonl",
        ]
        files = require_files(required)
        checkpoint = OUTPUTS / "catai-sft.pt"
        recovery = OUTPUTS / "catai-sft-recovery.pt"
        command = [
            sys.executable, "-m", "catai.sft_cli",
            "--base-checkpoint", str(files["catai.pt"]),
            "--dataset", str(files["oasst_chat.jsonl"]),
            "--personality-dataset", str(files["instruction_chat.jsonl"]),
            "--personality-ratio", "0.03",
            "--smol-smoltalk-dataset", str(files["smol_smoltalk_chat.jsonl"]),
            "--smol-smoltalk-ratio", "0.10",
            "--blended-skill-talk-dataset", str(files["blended_skill_talk_chat.jsonl"]),
            "--blended-skill-talk-ratio", "0.05",
            "--ultrachat-dataset", str(files["ultrachat_200k_chat.jsonl"]),
            "--ultrachat-ratio", "0.25",
            "--expanded-chat-dataset", str(files["expanded_sft_chat.jsonl"]),
            "--expanded-chat-ratio", "0.15",
            "--markdown-dataset", str(files["markdown_sft_chat.jsonl"]),
            "--markdown-ratio", "0.05",
            "--samples-per-epoch", "6000",
            "--max-validation-examples", "500",
            "--checkpoint", str(checkpoint),
            "--epochs", "3",
            "--batch-size", "2",
            "--learning-rate", "2e-5",
            "--minimum-tokens-per-parameter", "20",
            "--grad-clip", "1.0",
            "--validation-split", "0.05",
            "--patience", "2",
            "--sequence-length", "1024",
            "--seed", "42",
            "--device", "cuda",
            "--precision", "auto",
            "--recovery-checkpoint", str(recovery),
            "--checkpoint-every-batches", "50",
        ]
        if recovery.is_file():
            command.extend(["--resume-from", str(recovery)])

    import torch
    print("Starting CatAI", mode, "training on", torch.cuda.get_device_name(0), flush=True)
    try:
        subprocess.run(command, cwd="/root/catai", check=True)
    finally:
        # Persist recovery checkpoints even if the training command exits with an error.
        volume.commit()

    completed = checkpoint.is_file()
    if completed:
        bundle = OUTPUTS / ("catai-base.npz" if mode == "base" else "catai-sft.npz")
        subprocess.run(
            [
                sys.executable,
                "/root/catai/scripts/export_inference.py",
                "--checkpoint", str(checkpoint),
                "--output", str(bundle),
            ],
            check=True,
        )
    # Commit recovery state and completed artifacts before the GPU container exits.
    volume.commit()
    return {
        "completed": completed,
        "checkpoint": str(checkpoint),
        "recovery_checkpoint": str(recovery),
        "message": "Training completed." if completed else "Base training paused at a recovery checkpoint; run again to continue.",
    }


@app.local_entrypoint()
def main(
    mode: str = "sft",
    gpu: str = "L4",
    confirm_paid_compute: bool = False,
) -> None:
    """Launch Modal training. GPU compute is billable; acknowledgement is required."""
    if not confirm_paid_compute:
        raise SystemExit(
            "Modal GPU usage is billable. Review Modal pricing, then rerun with "
            "--confirm-paid-compute if you explicitly approve the charge."
        )
    if mode not in {"base", "sft"}:
        raise SystemExit("--mode must be base or sft")
    allowed_gpus = {"T4", "L4", "A10", "A100"}
    if gpu not in allowed_gpus:
        raise SystemExit(f"--gpu must be one of: {', '.join(sorted(allowed_gpus))}")

    print(f"Submitting CatAI {mode} training to Modal ({gpu}).")
    result = run_training.with_options(gpu=gpu).remote(mode)
    print(result)
    if mode == "base" and not result["completed"]:
        print("Base training is resumable. Re-run the same command to continue from the committed recovery checkpoint.")
