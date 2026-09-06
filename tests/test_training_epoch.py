import torch

from catai.batching import make_dataloader
from catai.dataset import TokenWindowDataset
from catai.model import CatAI
from catai.training import train_epoch


def test_train_epoch_consumes_all_batches():
    torch.manual_seed(0)
    model = CatAI(vocab_size=8, max_seq_len=4, d_model=16, n_heads=4, n_layers=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    dataset = TokenWindowDataset(torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]), sequence_length=4)
    loader = make_dataloader(dataset, batch_size=2, shuffle=False)

    loss = train_epoch(model, optimizer, loader)

    assert loss > 0
