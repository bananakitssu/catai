import torch

from catai.generation import generate
from catai.model import CatAI


def test_generate_appends_requested_number_of_tokens():
    torch.manual_seed(0)
    model = CatAI(vocab_size=12, max_seq_len=8, d_model=16, n_heads=4, n_layers=1)
    prompt = torch.tensor([[1, 2, 3]])
    result = generate(model, prompt, max_new_tokens=4)
    assert result.shape == (1, 7)
    assert torch.equal(result[:, :3], prompt)


def test_generate_handles_context_longer_than_model_window():
    torch.manual_seed(0)
    model = CatAI(vocab_size=12, max_seq_len=4, d_model=16, n_heads=4, n_layers=1)
    prompt = torch.tensor([[1, 2, 3, 4]])
    result = generate(model, prompt, max_new_tokens=2)
    assert result.shape == (1, 6)


def test_generate_validates_arguments():
    model = CatAI(vocab_size=8, max_seq_len=4, d_model=16, n_heads=4, n_layers=1)
    prompt = torch.tensor([[1, 2]])
    for kwargs in ({"max_new_tokens": -1}, {"max_new_tokens": 1, "temperature": 0}):
        try:
            generate(model, prompt, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("generate should reject invalid arguments")
