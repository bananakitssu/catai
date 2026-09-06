import torch

from catai.model import CatAI
from catai.training import causal_language_model_loss, make_next_token_batch, train_step


def test_next_token_batch_shifts_targets():
    tokens = torch.tensor([[1, 2, 3, 4]])
    inputs, targets = make_next_token_batch(tokens)
    assert inputs.tolist() == [[1, 2, 3]]
    assert targets.tolist() == [[2, 3, 4]]


def test_loss_is_scalar_and_differentiable():
    logits = torch.randn(2, 3, 8, requires_grad=True)
    targets = torch.randint(0, 8, (2, 3))
    loss = causal_language_model_loss(logits, targets)
    assert loss.ndim == 0
    loss.backward()
    assert logits.grad is not None


def test_training_step_updates_parameters():
    torch.manual_seed(0)
    model = CatAI(vocab_size=8, max_seq_len=8, d_model=16, n_heads=4, n_layers=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 1]])
    before = [parameter.detach().clone() for parameter in model.parameters()]
    train_step(model, optimizer, tokens)
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))
