import torch

from catai.model import CatAI
from catai.sft import (
    IGNORE_INDEX,
    ChatSupervisedDataset,
    chat_token_stream,
    supervised_causal_language_model_loss,
    sft_train_step,
)
from catai.tokenizer import CharTokenizer


def test_chat_token_stream_masks_non_assistant_tokens():
    tokenizer = CharTokenizer.default()
    tokens, mask = chat_token_stream(
        [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ],
        tokenizer,
    )

    assert len(tokens) == len(mask)
    assert any(mask)
    assert all(not enabled for enabled in mask[:10])


def test_chat_dataset_returns_fixed_length_masked_labels():
    tokenizer = CharTokenizer.default()
    dataset = ChatSupervisedDataset(
        [
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ]
        ],
        tokenizer,
        sequence_length=32,
    )

    sample = dataset[0]
    assert sample["input_ids"].shape == (32,)
    assert sample["labels"].shape == (32,)
    assert torch.any(sample["labels"] != IGNORE_INDEX)
    assert torch.any(sample["labels"] == IGNORE_INDEX)


def test_sft_loss_is_differentiable_with_ignored_targets():
    logits = torch.randn(2, 4, 8, requires_grad=True)
    labels = torch.tensor(
        [[IGNORE_INDEX, 2, 3, IGNORE_INDEX], [1, IGNORE_INDEX, 4, 5]]
    )

    loss = supervised_causal_language_model_loss(logits, labels)
    assert loss.ndim == 0
    loss.backward()
    assert logits.grad is not None


def test_sft_train_step_updates_parameters():
    torch.manual_seed(0)
    tokenizer = CharTokenizer.default()
    dataset = ChatSupervisedDataset(
        [
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ]
        ],
        tokenizer,
        sequence_length=12,
    )
    sample = dataset[0]

    model = CatAI(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=12,
        d_model=16,
        n_heads=4,
        n_layers=1,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    before = [parameter.detach().clone() for parameter in model.parameters()]

    sft_train_step(
        model,
        optimizer,
        sample["input_ids"].unsqueeze(0),
        sample["labels"].unsqueeze(0),
    )

    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, model.parameters())
    )
