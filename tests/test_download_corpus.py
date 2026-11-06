import gzip
import json

from data.download_corpus import filter_redpajama_shard, gopher_rules_pass


def good_signals():
    return {
        "rps_doc_word_count": [[0, 300, 60]],
        "rps_doc_mean_word_length": [[0, 300, 5.0]],
        "rps_doc_symbol_to_word_ratio": [[0, 300, 0.02]],
        "rps_doc_frac_chars_top_2gram": [[0, 300, 0.04]],
        "rps_lines_start_with_bulletpoint": [
            [0, 50, 0], [50, 100, 0], [100, 200, 0], [200, 300, 0]
        ],
    }


def test_gopher_filter_rejects_low_quality_signals():
    assert gopher_rules_pass(good_signals())
    noisy = dict(good_signals())
    noisy["rps_doc_symbol_to_word_ratio"] = [[0, 300, 0.2]]
    assert not gopher_rules_pass(noisy)
    short = dict(good_signals())
    short["rps_doc_word_count"] = [[0, 300, 12]]
    assert not gopher_rules_pass(short)


def test_redpajama_filter_drops_marked_and_exact_duplicates():
    text = (
        "This is a sufficiently long document with ordinary words in sentences. "
        "It continues with additional words to pass the document quality limits."
    )
    docs = [
        {"raw_content": text}, {"raw_content": text},
        {"raw_content": text + " A distinct ending."}, {"raw_content": "tiny"},
    ]
    signals = [
        {"id": f"{shard_path}/0", "quality_signals": good_signals()},
        {"id": f"{shard_path}/1", "quality_signals": good_signals()},
        {"id": f"{shard_path}/2", "quality_signals": good_signals()},
        {"id": f"{shard_path}/3", "quality_signals": good_signals()},
    ]
    doc_payload = gzip.compress("\n".join(json.dumps(x) for x in docs).encode())
    signal_payload = gzip.compress("\n".join(json.dumps(x) for x in signals).encode())
    stats = {}
    shard_path = "2023-06/0000/en_head_middle.json.gz"
    selected = filter_redpajama_shard(
        doc_payload, signal_payload, shard_path,
        {f"{shard_path}/2"}, set(), stats
    )
    assert selected == [text]
    assert stats["duplicate_ids"] == 1
    assert stats["exact_duplicates"] == 1
    assert stats["quality_rejected"] == 1



def test_redpajama_filter_joins_signals_by_id_with_missing_and_reordered_rows():
    texts = [
        "First valid document contains enough ordinary words to pass the quality filter.",
        "Second valid document is present but has no quality signal row in this shard.",
        "Third valid document contains enough ordinary words to pass the quality filter.",
    ]
    docs = [{"raw_content": text} for text in texts]
    shard_path = "2023-06/0000/en_head.json.gz"
    signals = [
        {"id": f"{shard_path}/2", "quality_signals": good_signals()},
        {"id": f"{shard_path}/0", "quality_signals": good_signals()},
    ]
    doc_payload = gzip.compress("\\n".join(json.dumps(x) for x in docs).encode())
    signal_payload = gzip.compress("\\n".join(json.dumps(x) for x in signals).encode())
    stats = {}

    selected = filter_redpajama_shard(
        doc_payload, signal_payload, shard_path, set(), set(), stats
    )

    assert selected == [texts[0], texts[2]]
    assert stats["seen"] == 3
    assert stats["missing_signals"] == 1
    assert stats["kept"] == 2
