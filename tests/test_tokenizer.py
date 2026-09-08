from catai.tokenizer import CharTokenizer, EOS_TOKEN


def test_encode_decode_round_trip() -> None:
    tokenizer = CharTokenizer.from_text("cat banana")
    text = "cat banana"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_vocabulary_is_deterministic() -> None:
    first = CharTokenizer.from_text("banana cat")
    second = CharTokenizer.from_text("cat banana")
    assert first.vocabulary == second.vocabulary
    assert first.vocab_size == len(set("banana cat")) + 1
    assert first.vocabulary[-1] == EOS_TOKEN


def test_legacy_vocabulary_without_eos_is_supported() -> None:
    tokenizer = CharTokenizer(("a", "b", "c"))
    assert tokenizer.eos_token_id is None
    assert tokenizer.vocab_size == 3
    assert tokenizer.decode(tokenizer.encode("abc")) == "abc"


def test_unknown_character_is_rejected() -> None:
    tokenizer = CharTokenizer.from_text("abc")
    try:
        tokenizer.encode("abd")
    except ValueError as exc:
        assert "unknown characters" in str(exc)
    else:
        raise AssertionError("expected ValueError")
