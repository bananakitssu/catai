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



class FixedLogitModel(torch.nn.Module):
    """Tiny deterministic model for testing generation filters."""

    def __init__(self, logits, context_length=16):
        super().__init__()
        self.register_buffer("fixed_logits", torch.tensor(logits, dtype=torch.float32))
        self.vocab_size = len(logits)
        self.context_length = context_length

    def forward(self, tokens):
        return self.fixed_logits.view(1, 1, -1).expand(tokens.size(0), tokens.size(1), -1)


def test_repetition_penalty_does_not_penalize_tokens_only_in_prompt():
    model = FixedLogitModel([0.0, 2.0, 1.9, 0.0])
    prompt = torch.tensor([[1, 3]])

    result = generate(
        model,
        prompt,
        max_new_tokens=1,
        temperature=1.0,
        top_k=1,
        repetition_penalty=1.1,
        no_repeat_ngram_size=0,
    )

    # Token 1 is present in the prompt but should not be penalized before
    # the model has generated any reply tokens.
    assert result[0, -1].item() == 1


def test_no_repeat_ngram_does_not_block_ngrams_only_in_prompt():
    model = FixedLogitModel([0.0, 0.0, 0.0, 5.0])
    prompt = torch.tensor([[1, 2, 3]])

    result = generate(
        model,
        prompt,
        max_new_tokens=1,
        temperature=1.0,
        top_k=1,
        repetition_penalty=1.0,
        no_repeat_ngram_size=3,
    )

    # The first generated character may complete an n-gram seen in the
    # prompt; only repetition within the reply should be blocked.
    assert result[0, -1].item() == 3
