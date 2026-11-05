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
        {"id": "keep", "quality_signals": good_signals()},
        {"id": "same-text", "quality_signals": good_signals()},
        {"id": "remote-duplicate", "quality_signals": good_signals()},
        {"id": "short", "quality_signals": good_signals()},
    ]
    doc_payload = gzip.compress("\n".join(json.dumps(x) for x in docs).encode())
    signal_payload = gzip.compress("\n".join(json.dumps(x) for x in signals).encode())
    stats = {}
    selected = filter_redpajama_shard(
        doc_payload, signal_payload, {"remote-duplicate"}, set(), stats
    )
    assert selected == [text]
    assert stats["duplicate_ids"] == 1
    assert stats["exact_duplicates"] == 1
    assert stats["quality_rejected"] == 1
