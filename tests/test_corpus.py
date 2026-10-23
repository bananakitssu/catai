from catai.corpus import tokenizer_and_tokens


def test_bpe_corpus_loader(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_text("banana cat banana", encoding="utf-8")
    tokenizer, tokens = tokenizer_and_tokens(path, tokenizer_type="bpe", vocab_size=12)
    assert tokenizer.vocab_size <= 12
    assert tokens.dtype.name == "int64" if hasattr(tokens.dtype, "name") else str(tokens.dtype) == "torch.int64"
    assert tokenizer.decode(tokens.tolist()) == "banana cat banana"


def test_corpus_token_budget(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_text("abcdefghij", encoding="utf-8")
    tokenizer, tokens = tokenizer_and_tokens(path, max_tokens=4)
    assert len(tokens) == 5
    assert tokens[-1].item() == tokenizer.eos_token_id



def test_bpe_training_sample_still_encodes_full_corpus(tmp_path):
    path = tmp_path / "corpus.txt"
    text = "a" * 100 + " rare-symbol Ω"
    path.write_text(text, encoding="utf-8")
    tokenizer, tokens = tokenizer_and_tokens(
        path,
        tokenizer_type="bpe",
        vocab_size=16,
        tokenizer_sample_chars=20,
    )
    assert tokenizer.decode(tokens.tolist()) == text


def test_token_budget_does_not_encode_all_bpe_tokens(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_text("banana cat " * 100, encoding="utf-8")
    _, tokens = tokenizer_and_tokens(
        path,
        tokenizer_type="bpe",
        vocab_size=16,
        max_tokens=5,
        tokenizer_sample_chars=40,
    )
    assert len(tokens) == 6
