from pathlib import Path

import torch

from catai.checkpoint import load_checkpoint, load_training_metadata, save_checkpoint
from catai.model import CatAI
from catai.training import train_epochs


def test_checkpoint_can_resume_training(tmp_path: Path) -> None:
    torch.manual_seed(0)
    tokens = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
    model = CatAI(vocab_size=4, max_seq_len=7, d_model=16, n_heads=4, n_layers=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    train_epochs(model, optimizer, torch.utils.data.DataLoader(tokens, batch_size=1), epochs=1)
    checkpoint = tmp_path / "checkpoints" / "resume.pt"
    save_checkpoint(checkpoint, model, optimizer, step=8, epoch=1, loss=0.5)

    resumed = CatAI(vocab_size=4, max_seq_len=7, d_model=16, n_heads=4, n_layers=1)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-2)
    step = load_checkpoint(checkpoint, resumed, resumed_optimizer)
    metadata = load_training_metadata(checkpoint)

    assert step == 8
    assert metadata == {"step": 8, "epoch": 1, "loss": 0.5}

    before = [parameter.detach().clone() for parameter in resumed.parameters()]
    train_epochs(resumed, resumed_optimizer, torch.utils.data.DataLoader(tokens, batch_size=1), epochs=1)
    assert any(not torch.equal(old, new) for old, new in zip(before, resumed.parameters()))
