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
