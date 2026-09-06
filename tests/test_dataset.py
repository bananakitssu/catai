import pytest
import torch

from catai.dataset import TokenWindowDataset


def test_token_window_dataset_creates_overlapping_windows():
    tokens = torch.tensor([0, 1, 2, 3, 4])
    dataset = TokenWindowDataset(tokens, sequence_length=3)

    assert len(dataset) == 2
    assert dataset[0].tolist() == [0, 1, 2, 3]
    assert dataset[1].tolist() == [1, 2, 3, 4]


def test_token_window_dataset_validates_inputs():
    with pytest.raises(ValueError):
        TokenWindowDataset(torch.tensor([[1, 2, 3]]), sequence_length=2)
    with pytest.raises(ValueError):
        TokenWindowDataset(torch.tensor([1, 2, 3]), sequence_length=0)
    with pytest.raises(ValueError):
        TokenWindowDataset(torch.tensor([1, 2]), sequence_length=2)
