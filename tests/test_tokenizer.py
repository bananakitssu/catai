from catai.tokenizer import CharTokenizer


def test_encode_decode_round_trip() -> None:
    tokenizer = CharTokenizer.from_text("cat banana")
    text = "cat banana"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_vocabulary_is_deterministic() -> None:
    first = CharTokenizer.from_text("banana cat")
    second = CharTokenizer.from_text("cat banana")
    assert first.vocabulary == second.vocabulary
    assert first.vocab_size == len(set("banana cat"))


def test_unknown_character_is_rejected() -> None:
    tokenizer = CharTokenizer.from_text("abc")
    try:
        tokenizer.encode("abd")
    except ValueError as exc:
        assert "unknown characters" in str(exc)
    else:
        raise AssertionError("expected ValueError")
