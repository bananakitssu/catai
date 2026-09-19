import torch

from catai.model import CatAI, RMSNorm, SwiGLU


def test_model_forward_shape():
    model = CatAI(vocab_size=64, max_seq_len=32, d_model=64, n_heads=4, n_layers=2)
    tokens = torch.randint(0, 64, (2, 16))
    logits = model(tokens)
    assert logits.shape == (2, 16, 64)


def test_model_parameter_count_positive():
    model = CatAI(vocab_size=100, max_seq_len=64, d_model=128, n_heads=4, n_layers=4)
    assert model.parameter_count() > 0


def test_rmsnorm_preserves_shape():
    norm = RMSNorm(32)
    x = torch.randn(2, 10, 32)
    y = norm(x)
    assert y.shape == x.shape


def test_swiglu_preserves_shape():
    mlp = SwiGLU(64)
    x = torch.randn(2, 8, 64)
    y = mlp(x)
    assert y.shape == x.shape


def test_longer_context_works():
    model = CatAI(vocab_size=50, max_seq_len=128, d_model=64, n_heads=4, n_layers=2)
    tokens = torch.randint(0, 50, (1, 100))
    logits = model(tokens)
    assert logits.shape == (1, 100, 50)
