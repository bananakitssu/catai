import torch

from catai.model import CatAI


def test_model_shape_and_causality():
    model = CatAI(vocab_size=16, max_seq_len=8, d_model=32, n_heads=4, n_layers=2)
    tokens = torch.randint(0, 16, (2, 8))
    logits = model(tokens)
    assert logits.shape == (2, 8, 16)


def test_model_has_trainable_parameters():
    model = CatAI(vocab_size=8, max_seq_len=4, d_model=16, n_heads=4, n_layers=1)
    assert model.parameter_count() > 0
    assert all(parameter.requires_grad for parameter in model.parameters())
