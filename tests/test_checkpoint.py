import torch

from catai.checkpoint import load_checkpoint, save_checkpoint
from catai.model import CatAI


def test_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)
    model = CatAI(vocab_size=8, max_seq_len=8, d_model=16, n_heads=4, n_layers=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    inputs = torch.tensor([[0, 1, 2, 3]])

    loss = model(inputs).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, step=7)

    restored = CatAI(vocab_size=8, max_seq_len=8, d_model=16, n_heads=4, n_layers=1)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-2)
    step = load_checkpoint(path, restored, restored_optimizer)

    assert step == 7
    for name, value in restored.state_dict().items():
        assert torch.equal(value, expected[name])
    assert restored_optimizer.state_dict()["state"] == optimizer.state_dict()["state"]


def test_checkpoint_rejects_negative_step(tmp_path):
    torch.manual_seed(0)
    model = CatAI(vocab_size=4, max_seq_len=4, d_model=8, n_heads=2, n_layers=1)
    optimizer = torch.optim.AdamW(model.parameters())

    try:
        save_checkpoint(tmp_path / "checkpoint.pt", model, optimizer, step=-1)
    except ValueError as exc:
        assert str(exc) == "step must be non-negative"
    else:
        raise AssertionError("negative step should be rejected")
