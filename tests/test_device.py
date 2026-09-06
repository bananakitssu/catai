import torch

from catai.device import resolve_device


def test_resolve_device_defaults_to_available_backend() -> None:
    device = resolve_device()
    assert device.type in {"cpu", "cuda", "mps"}


def test_resolve_device_accepts_explicit_device() -> None:
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device(torch.device("cpu")) == torch.device("cpu")
