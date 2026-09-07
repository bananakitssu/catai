import torch

from catai.checkpoint import load_checkpoint, load_checkpoint_metadata, save_checkpoint
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

    original_state = optimizer.state_dict()["state"]
    restored_state = restored_optimizer.state_dict()["state"]
    assert restored_state.keys() == original_state.keys()
    for parameter_id in original_state:
        for key, value in original_state[parameter_id].items():
            restored_value = restored_state[parameter_id][key]
            if torch.is_tensor(value):
                assert torch.equal(restored_value, value)
            else:
                assert restored_value == value


def test_checkpoint_metadata_round_trip(tmp_path):
    model = CatAI(vocab_size=4, max_seq_len=8, d_model=8, n_heads=2, n_layers=1)
    optimizer = torch.optim.AdamW(model.parameters())
    metadata = {
        "model": {"max_seq_len": 8, "d_model": 8, "n_heads": 2, "n_layers": 1},
        "tokenizer": {"type": "char", "vocabulary": ["a", "b", "c", "d"]},
    }
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, step=3, metadata=metadata)

    assert load_checkpoint_metadata(path) == metadata


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
