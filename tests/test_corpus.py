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


def test_bpe_corpus_vocab_stays_within_the_requested_budget(tmp_path):
    rare_characters = "".join(chr(0x1000 + index) for index in range(600))
    path = tmp_path / "unicode-corpus.txt"
    path.write_text(("the cat sits. " * 1000) + rare_characters, encoding="utf-8")

    tokenizer, tokens = tokenizer_and_tokens(
        path,
        tokenizer_type="bpe",
        vocab_size=512,
        tokenizer_sample_chars=5000,
    )

    assert tokenizer.vocab_size <= 512
    assert tokenizer.unk_token_id is not None
    assert tokenizer.unk_token_id in tokens.tolist()


def test_corpus_loader_preserves_explicit_document_end_tokens(tmp_path):
    from catai.tokenizer import EOS_TOKEN

    path = tmp_path / "documents.txt"
    path.write_text("first document" + EOS_TOKEN + "\nsecond document", encoding="utf-8")
    tokenizer, tokens = tokenizer_and_tokens(
        path,
        tokenizer_type="bpe",
        vocab_size=64,
        max_tokens=100,
    )

    eos_id = tokenizer.eos_token_id
    eos_positions = (tokens == eos_id).nonzero(as_tuple=True)[0].tolist()
    assert len(eos_positions) == 2  # inter-document EOS and final corpus EOS
    assert eos_positions[0] == len(tokenizer.encode("first document"))
