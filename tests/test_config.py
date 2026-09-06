import pytest

from catai.config import TrainingConfig


def test_training_config_defaults() -> None:
    config = TrainingConfig()
    assert config.batch_size == 32
    assert config.learning_rate == 3e-4
    assert config.epochs == 1
    assert config.grad_clip == 1.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 0},
        {"learning_rate": 0},
        {"epochs": 0},
        {"grad_clip": 0},
    ],
)
def test_training_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        TrainingConfig(**kwargs)


def test_training_config_allows_disabling_gradient_clipping() -> None:
    assert TrainingConfig(grad_clip=None).grad_clip is None
