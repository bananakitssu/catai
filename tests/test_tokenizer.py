from catai.tokenizer import BPETokenizer, EOS_TOKEN, UNK_TOKEN


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
