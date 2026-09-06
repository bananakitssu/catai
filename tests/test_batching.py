import pytest
import torch

from catai.batching import make_dataloader
from catai.dataset import TokenWindowDataset


def test_dataloader_batches_token_windows():
    dataset = TokenWindowDataset(torch.arange(10), sequence_length=3)
    loader = make_dataloader(dataset, batch_size=2, shuffle=False)

    batch = next(iter(loader))
    assert batch.shape == (2, 4)
    assert batch.tolist() == [[0, 1, 2, 3], [1, 2, 3, 4]]


def test_dataloader_validates_batch_size():
    dataset = TokenWindowDataset(torch.arange(5), sequence_length=2)
    with pytest.raises(ValueError):
        make_dataloader(dataset, batch_size=0)
