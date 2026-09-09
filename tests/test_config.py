from catai.config import MODEL_PRESETS, ModelConfig, TrainingConfig


def test_training_config_rejects_invalid_values():
    for kwargs in ({"batch_size": 0}, {"learning_rate": 0}, {"epochs": 0}, {"grad_clip": 0}, {"validation_split": 1}, {"patience": 0}, {"max_tokens": 0}):
        try:
            TrainingConfig(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {kwargs}")


def test_model_config_validates_dimensions():
    assert ModelConfig() == ModelConfig(128, 4, 4)
    try:
        ModelConfig(d_model=10, n_heads=3)
    except ValueError:
        pass
    else:
        raise AssertionError("expected divisibility validation")


def test_model_presets_are_scalable():
    tiny = ModelConfig.from_preset("tiny")
    small = ModelConfig.from_preset("small")
    base = ModelConfig.from_preset("base")
    assert tiny.n_layers < small.n_layers < base.n_layers
    assert tiny.d_model < small.d_model < base.d_model
    try:
        ModelConfig.from_preset("unknown")
    except ValueError:
        pass
    else:
        raise AssertionError("expected unknown preset to fail")
