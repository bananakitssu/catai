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



def test_streaming_corpus_encoder_matches_whole_text_bpe_encoding(tmp_path):
    from catai.corpus import _encode_prefix_tensor_from_file
    from catai.tokenizer import BPETokenizer, EOS_TOKEN

    text = (
        "A banana is yellow and ripe.\n"
        + EOS_TOKEN
        + "\n"
        + "A second document contains some more ordinary words.\n"
    )
    tokenizer = BPETokenizer.from_text(text, vocab_size=64)
    path = tmp_path / "streamed.txt"
    path.write_text(text, encoding="utf-8")

    expected = tokenizer.encode(text)
    if expected[-1] != tokenizer.eos_token_id:
        expected.append(tokenizer.eos_token_id)

    actual = _encode_prefix_tensor_from_file(
        tokenizer, path, None, chunk_chars=11
    )
    assert actual.tolist() == expected


def test_tokenizer_and_tokens_does_not_load_the_entire_corpus(tmp_path, monkeypatch):
    import catai.corpus as corpus_module
    from catai.tokenizer import EOS_TOKEN

    path = tmp_path / "streamed-documents.txt"
    path.write_text(
        ("A banana is yellow.\n" + EOS_TOKEN + "\n") * 50,
        encoding="utf-8",
    )

    def reject_full_text_load(*args, **kwargs):
        raise AssertionError("tokenizer_and_tokens must stream the corpus")

    monkeypatch.setattr(corpus_module, "load_text", reject_full_text_load)
    tokenizer, tokens = corpus_module.tokenizer_and_tokens(
        path,
        tokenizer_type="bpe",
        vocab_size=64,
        max_tokens=100,
        tokenizer_sample_chars=32,
    )
    assert tokenizer.eos_token_id is not None
    assert 100 <= tokens.numel() <= 101
    assert tokens[-1].item() == tokenizer.eos_token_id



def test_corpus_loader_reuses_checkpoint_tokenizer_without_relearning(tmp_path, monkeypatch):
    import catai.corpus as corpus_module
    from catai.tokenizer import BPETokenizer, EOS_TOKEN

    text = "The cat learns new words. " * 20 + EOS_TOKEN
    path = tmp_path / "resume-corpus.txt"
    path.write_text(text, encoding="utf-8")
    tokenizer = BPETokenizer.from_text(text, vocab_size=64)

    def fail_tokenizer_scan(*args, **kwargs):
        raise AssertionError("resume must not relearn the checkpoint tokenizer")

    monkeypatch.setattr(corpus_module, "_read_tokenizer_inputs", fail_tokenizer_scan)
    loaded_tokenizer, tokens = corpus_module.tokenizer_and_tokens(
        path,
        tokenizer_type="bpe",
        vocab_size=64,
        tokenizer=tokenizer,
    )

    assert loaded_tokenizer == tokenizer
    assert tokenizer.decode(tokens.tolist()) == text
