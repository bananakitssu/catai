import torch
from torch import nn

from catai.training import clip_gradients


def test_clip_gradients_limits_global_norm() -> None:
    model = nn.Linear(2, 1)
    model.weight.grad = torch.full_like(model.weight, 10.0)
    model.bias.grad = torch.full_like(model.bias, 10.0)

    before = clip_gradients(model, 0.5)

    after = torch.linalg.vector_norm(
        torch.cat([parameter.grad.reshape(-1) for parameter in model.parameters() if parameter.grad is not None])
    )
    assert before > 0.5
    assert float(after) <= 0.5 + 1e-6
