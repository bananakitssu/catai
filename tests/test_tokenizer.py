from catai.tokenizer import BPETokenizer, EOS_TOKEN


def test_bpe_round_trip():
    tokenizer = BPETokenizer.from_text("banana banana cat", vocab_size=16)
    text = "banana cat"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_bpe_has_merges_and_eos():
    tokenizer = BPETokenizer.from_text("banana banana banana", vocab_size=10)
    assert tokenizer.eos_token_id == tokenizer.vocab_size - 1
    assert tokenizer.vocabulary[-1] == EOS_TOKEN
    assert tokenizer.merges


def test_bpe_is_deterministic():
    first = BPETokenizer.from_text("banana cat banana", vocab_size=12)
    second = BPETokenizer.from_text("banana cat banana", vocab_size=12)
    assert first == second
