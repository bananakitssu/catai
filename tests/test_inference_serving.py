from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inference"))
sys.path.insert(0, str(ROOT / "scripts"))

from catai.checkpoint import save_checkpoint
from catai.model import CatAI
from catai.tokenizer import CharTokenizer
from export_inference import export_checkpoint
import runtime as inference_runtime


def _metadata(model: CatAI) -> dict:
    tokenizer = CharTokenizer.default()
    return {
        "model": {
            "max_seq_len": model.context_length,
            "d_model": model.d_model,
            "n_heads": model.n_heads,
            "n_layers": model.n_layers,
        },
        "tokenizer": {
            "type": "char",
            "vocabulary": list(tokenizer.vocabulary),
        },
    }


def test_exporter_strips_optimizer_state_and_writes_safe_bundle(tmp_path: Path) -> None:
    tokenizer = CharTokenizer.default()
    model = CatAI(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=12,
        d_model=16,
        n_heads=4,
        n_layers=2,
    )
    optimizer = torch.optim.AdamW(model.parameters())
    checkpoint = tmp_path / "training.pt"
    output = tmp_path / "inference.npz"
    save_checkpoint(
        checkpoint,
        model,
        optimizer,
        step=7,
        metadata=_metadata(model),
    )

    export_checkpoint(checkpoint, output)

    with np.load(output, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["__metadata__"].item()))
        assert metadata["format"] == "catai-numpy-inference"
        assert metadata["step"] == 7
        assert metadata["model"]["max_seq_len"] == 12
        assert "optimizer_state_dict" not in archive.files
        assert "token_embedding.weight" in archive.files
        # Input/output embeddings are tied in CatAI; the exporter omits the duplicate.
        assert "lm_head.weight" not in archive.files


def test_numpy_forward_matches_pytorch_reference() -> None:
    torch.manual_seed(4)
    tokenizer = CharTokenizer.default()
    model = CatAI(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=16,
        d_model=16,
        n_heads=4,
        n_layers=2,
    ).eval()
    weights = {
        name: value.detach().cpu().numpy().astype(np.float32, copy=False)
        for name, value in model.state_dict().items()
    }
    tokens = [4, 18, 31, 42, 8, 2]
    with torch.no_grad():
        expected = model(torch.tensor([tokens], dtype=torch.long))[0, -1].cpu().numpy()

    actual = inference_runtime._forward(
        weights,
        {
            "max_seq_len": 16,
            "d_model": 16,
            "n_heads": 4,
            "n_layers": 2,
        },
        tokens,
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5)
