import torch

from catai.batching import make_dataloader
from catai.dataset import TokenWindowDataset
from catai.model import CatAI
from catai.training import train_epochs


def test_train_epochs_reduces_loss_on_repeated_windows():
    torch.manual_seed(0)
    tokens = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3] * 4)
    dataset = TokenWindowDataset(tokens, sequence_length=7)
    loader = make_dataloader(dataset, batch_size=8, shuffle=False)
    model = CatAI(vocab_size=4, max_seq_len=7, d_model=16, n_heads=4, n_layers=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    losses = train_epochs(model, optimizer, loader, epochs=4)

    assert len(losses) == 4
    assert losses[-1] < losses[0]


def test_train_epochs_rejects_non_positive_epochs():
    model = CatAI(vocab_size=4, max_seq_len=4, d_model=8, n_heads=2, n_layers=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    tokens = torch.tensor([0, 1, 2, 3, 0])
    loader = make_dataloader(TokenWindowDataset(tokens, sequence_length=4), batch_size=1)

    try:
        train_epochs(model, optimizer, loader, epochs=0)
    except ValueError as exc:
        assert "epochs" in str(exc)
    else:
        raise AssertionError("expected ValueError")
