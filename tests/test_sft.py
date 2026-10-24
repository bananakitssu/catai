import pytest

import torch

from catai.chat_dataset import ASSISTANT_OUTPUT_ROLES, DEFAULT_STATE, state_system_message
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


@pytest.mark.parametrize("role", sorted(ASSISTANT_OUTPUT_ROLES))
def test_chat_token_stream_trains_extended_assistant_roles(role):
    tokenizer = CharTokenizer.default()
    tokens, mask = chat_token_stream(
        [
            {"role": "user", "content": "Hello"},
            {"role": role, "content": "generated output"},
        ],
        tokenizer,
    )

    assert len(tokens) == len(mask)
    assert any(mask)
    user_prefix = tokenizer.encode("<|user|>\nHello\n")
    assert all(not enabled for enabled in mask[: len(user_prefix)])


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
        sequence_length=40,
    )

    sample = dataset[0]
    assert sample["input_ids"].shape == (40,)
    assert sample["labels"].shape == (40,)
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
        sequence_length=40,
    )
    sample = dataset[0]

    model = CatAI(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=40,
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


def test_chat_dataset_drops_old_turns_without_splitting_latest_question_and_answer():
    tokenizer = CharTokenizer.default()
    dataset = ChatSupervisedDataset(
        [
            [
                {"role": "system", "content": "RULES"},
                {"role": "user", "content": "What was the old question?"},
                {"role": "assistant", "content": "This was the old answer."},
                {"role": "user", "content": "Explain recursion in one sentence."},
                {"role": "assistant", "content": "Recursion is when a function calls itself."},
            ]
        ],
        tokenizer,
        sequence_length=128,
    )

    sample = dataset[0]
    decoded = tokenizer.decode(sample["input_ids"].tolist())
    assert dataset.compacted_examples == 1
    assert dataset.skipped_overlength_examples == 0
    assert "RULES" in decoded
    assert "What was the old question?" not in decoded
    assert "Explain recursion in one sentence." in decoded
    assert "Recursion is when a function calls itself." in decoded
    assert torch.any(sample["labels"] != IGNORE_INDEX)


def test_chat_dataset_skips_a_latest_turn_that_cannot_fit_whole():
    tokenizer = CharTokenizer.default()
    messages = [
        {"role": "user", "content": "Question " + "extra details " * 80},
        {"role": "assistant", "content": "The answer starts here and cannot be fitted whole."},
    ]
    with pytest.raises(ValueError, match="no chat examples fit"):
        ChatSupervisedDataset([messages], tokenizer, sequence_length=64)


def test_chat_dataset_loads_and_validates_structured_state(tmp_path):
    from catai.chat_dataset import load_chat_dataset

    state = {
        "emotions": {
            "happiness": 0.7, "sadness": 0.1, "affection": 0.9,
            "curiosity": 0.8, "excitement": 0.6, "frustration": 0.1,
            "anger": 0.0, "fear": 0.0, "calmness": 0.5,
            "confidence": 0.8, "loneliness": 0.1, "playfulness": 0.9,
        },
        "needs": {
            "social_connection": 0.8, "stimulation": 0.7,
            "task_completion": 0.6, "rest": 0.2,
        },
        "personality": {
            "playful": 0.9, "curious": 0.85, "helpful": 0.8,
            "affectionate": 0.9, "seriousness": 0.3,
        },
    }
    import json

    path = tmp_path / "state.jsonl"
    path.write_text(
        json.dumps({
            "state": state,
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hiii! :3"},
            ],
        }) + "\n",
        encoding="utf-8",
    )

    examples = load_chat_dataset(path)
    assert examples[0][0]["role"] == "system"
    assert examples[0][0]["content"].startswith("You are CatAI, a friendly cat-themed conversational AI.")
    assert "CatAI internal state:" in examples[0][0]["content"]
    assert '"happiness":0.7' in examples[0][0]["content"]


def test_chat_dataset_rejects_out_of_range_state(tmp_path):
    from catai.chat_dataset import load_chat_dataset

    state = {
        "emotions": {
            "happiness": 1.2, "sadness": 0.1, "affection": 0.9,
            "curiosity": 0.8, "excitement": 0.6, "frustration": 0.1,
            "anger": 0.0, "fear": 0.0, "calmness": 0.5,
            "confidence": 0.8, "loneliness": 0.1, "playfulness": 0.9,
        },
        "needs": {
            "social_connection": 0.8, "stimulation": 0.7,
            "task_completion": 0.6, "rest": 0.2,
        },
        "personality": {
            "playful": 0.9, "curious": 0.85, "helpful": 0.8,
            "affectionate": 0.9, "seriousness": 0.3,
        },
    }
    import json

    path = tmp_path / "bad-state.jsonl"
    path.write_text(
        json.dumps({
            "state": state,
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ],
        }) + "\n",
        encoding="utf-8",
    )

    import pytest
    with pytest.raises(ValueError, match="between 0 and 1"):
        load_chat_dataset(path)



def test_chat_dataset_requires_state(tmp_path):
    from catai.chat_dataset import load_chat_dataset

    import json

    path = tmp_path / "missing-state.jsonl"
    path.write_text(
        json.dumps({
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ],
        }) + "\n",
        encoding="utf-8",
    )

    import pytest
    with pytest.raises(ValueError, match="missing required state"):
        load_chat_dataset(path)


def test_chat_dataset_preserves_state_system_prompt_when_truncated():
    tokenizer = CharTokenizer.default()
    messages = [
        state_system_message(DEFAULT_STATE),
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hiii! :3"},
    ]
    dataset = ChatSupervisedDataset(
        [messages],
        tokenizer,
        sequence_length=768,
    )

    sample = dataset[0]
    expected_prefix = tokenizer.encode(
        "<|system|>\n" + state_system_message(DEFAULT_STATE)["content"] + "\n"
    )
    assert sample["input_ids"][: len(expected_prefix)].tolist() == expected_prefix
    assert torch.any(sample["labels"] != IGNORE_INDEX)
