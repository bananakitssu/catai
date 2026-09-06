from pathlib import Path

import torch

from catai.batching import make_dataloader
from catai.config import TrainingConfig
from catai.corpus import tokenizer_and_tokens
from catai.dataset import TokenWindowDataset
from catai.model import CatAI
from catai.training import train_epochs


def test_tiny_corpus_training_reduces_loss() -> None:
    torch.manual_seed(0)
    root = Path(__file__).resolve().parents[1]
    tokenizer, tokens = tokenizer_and_tokens(root / "data" / "tiny.txt")
    dataset = TokenWindowDataset(tokens, sequence_length=16)
    loader = make_dataloader(dataset, batch_size=8, shuffle=False)

    model = CatAI(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=16,
        d_model=32,
        n_heads=4,
        n_layers=1,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    config = TrainingConfig(batch_size=8, learning_rate=3e-3, epochs=3, grad_clip=1.0)

    losses = train_epochs(
        model,
        optimizer,
        loader,
        epochs=config.epochs,
        grad_clip=config.grad_clip,
    )

    assert len(losses) == config.epochs
    assert all(torch.isfinite(torch.tensor(loss)) for loss in losses)
    assert losses[-1] < losses[0]
