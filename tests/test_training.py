import torch

from catai.model import CatAI
from catai.training import (
    causal_language_model_loss,
    make_next_token_batch,
    train,
    train_step,
)


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


def test_training_loop_reduces_loss():
    torch.manual_seed(0)
    model = CatAI(vocab_size=8, max_seq_len=16, d_model=16, n_heads=4, n_layers=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    tokens = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7] * 2])

    inputs, targets = make_next_token_batch(tokens)
    with torch.no_grad():
        initial_loss = causal_language_model_loss(model(inputs), targets).item()

    losses = train(model, optimizer, tokens, steps=40)

    assert len(losses) == 40
    assert losses[-1] < initial_loss


def test_cosine_warmup_learning_rate_starts_at_warmup_floor_and_reaches_peak():
    from catai.training import cosine_warmup_learning_rate

    assert cosine_warmup_learning_rate(
        1.0, 0, 100, warmup_fraction=0.1, minimum_learning_rate_ratio=0.2
    ) == 0.1
    assert cosine_warmup_learning_rate(
        1.0, 9, 100, warmup_fraction=0.1, minimum_learning_rate_ratio=0.2
    ) == 1.0


def test_cosine_warmup_learning_rate_decays_to_configured_floor():
    from catai.training import cosine_warmup_learning_rate

    final = cosine_warmup_learning_rate(
        2e-4, 99, 100, warmup_fraction=0.1, minimum_learning_rate_ratio=0.25
    )
    assert abs(final - 5e-5) < 1e-12


def test_cosine_warmup_learning_rate_validates_inputs():
    import pytest
    from catai.training import cosine_warmup_learning_rate

    with pytest.raises(ValueError):
        cosine_warmup_learning_rate(1.0, 0, 0)
    with pytest.raises(ValueError):
        cosine_warmup_learning_rate(1.0, 0, 10, warmup_fraction=1.0)
