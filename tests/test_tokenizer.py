from catai.tokenizer import BPETokenizer, CharTokenizer, EOS_TOKEN, UNK_TOKEN, PRINTABLE_ASCII


def test_char_fixed_vocab_can_spell_unseen_names():
    """A name that never appeared in training text must still be encodable."""
    tokenizer = CharTokenizer.from_text("the cat sat on the mat")
    name = "BTDPE"
    encoded = tokenizer.encode(name)
    assert tokenizer.decode(encoded) == name
    # Every character should be a real token, not UNK.
    assert tokenizer.unk_token_id not in encoded


def test_char_default_covers_printable_ascii():
    tokenizer = CharTokenizer.default()
    for ch in PRINTABLE_ASCII:
        assert ch in tokenizer.vocabulary
    assert tokenizer.vocabulary[-2:] == (UNK_TOKEN, EOS_TOKEN)


def test_char_unknown_unicode_becomes_unk():
    tokenizer = CharTokenizer.default()
    encoded = tokenizer.encode("hello 🐈")
    assert tokenizer.unk_token_id is not None
    assert tokenizer.unk_token_id in encoded
    assert tokenizer.decode(encoded).endswith("�")


def test_bpe_round_trip():
    tokenizer = BPETokenizer.from_text("banana banana cat", vocab_size=16)
    text = "banana cat"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_bpe_round_trip_repeated_training_text():
    tokenizer = BPETokenizer.from_text("banana cat banana", vocab_size=12)
    text = "banana cat banana"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_bpe_unknown_piece_uses_unk():
    tokenizer = BPETokenizer.from_text("banana cat", vocab_size=8)
    encoded = tokenizer.encode("banana 🐈")
    assert tokenizer.unk_token_id is not None
    assert encoded[-1] == tokenizer.unk_token_id
    assert tokenizer.decode(encoded).endswith("�")


def test_bpe_has_merges_and_special_tokens():
    tokenizer = BPETokenizer.from_text("banana banana banana", vocab_size=10)
    assert tokenizer.unk_token_id == tokenizer.vocab_size - 2
    assert tokenizer.eos_token_id == tokenizer.vocab_size - 1
    assert tokenizer.vocabulary[-2:] == (UNK_TOKEN, EOS_TOKEN)
    assert tokenizer.merges


def test_bpe_is_deterministic():
    first = BPETokenizer.from_text("banana cat banana", vocab_size=12)
    second = BPETokenizer.from_text("banana cat banana", vocab_size=12)
    assert first == second
