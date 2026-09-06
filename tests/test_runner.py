import torch

from catai.batching import make_dataloader
from catai.config import TrainingConfig
from catai.dataset import TokenWindowDataset
from catai.model import CatAI
from catai.runner import run_training


def test_run_training_uses_config_and_returns_losses() -> None:
    torch.manual_seed(0)
    tokens = torch.tensor([0, 1, 2, 3] * 8)
    loader = make_dataloader(TokenWindowDataset(tokens, sequence_length=7), batch_size=8, shuffle=False)
    model = CatAI(vocab_size=4, max_seq_len=7, d_model=16, n_heads=4, n_layers=1)
    config = TrainingConfig(batch_size=8, learning_rate=1e-2, epochs=3, grad_clip=1.0)

    losses = run_training(model, loader, config)

    assert len(losses) == 3
    assert losses[-1] < losses[0]
