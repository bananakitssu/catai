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
    shard_path = "2023-06/0000/en_head_middle.json.gz"
    signals = [
        {"id": f"{shard_path}/0", "quality_signals": good_signals()},
        {"id": f"{shard_path}/1", "quality_signals": good_signals()},
        {"id": f"{shard_path}/2", "quality_signals": good_signals()},
        {"id": f"{shard_path}/3", "quality_signals": good_signals()},
    ]
    doc_payload = gzip.compress("\n".join(json.dumps(x) for x in docs).encode())
    signal_payload = gzip.compress("\n".join(json.dumps(x) for x in signals).encode())
    stats = {}
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
    doc_payload = gzip.compress("\n".join(json.dumps(x) for x in docs).encode())
    signal_payload = gzip.compress("\n".join(json.dumps(x) for x in signals).encode())
    stats = {}

    selected = filter_redpajama_shard(
        doc_payload, signal_payload, shard_path, set(), set(), stats
    )

    assert selected == [texts[0], texts[2]]
    assert stats["seen"] == 3
    assert stats["missing_signals"] == 1
    assert stats["kept"] == 2


def test_each_pretraining_document_is_followed_by_an_eos_marker():
    from io import StringIO

    from data.download_corpus import DOCUMENT_EOS_MARKER, _write_document

    output = StringIO()
    _write_document(output, "First document text.")
    _write_document(output, "Second document text.")

    assert output.getvalue() == (
        "First document text.\n" + DOCUMENT_EOS_MARKER + "\n\n"
        "Second document text.\n" + DOCUMENT_EOS_MARKER + "\n\n"
    )



def test_common_corpus_license_filter_is_conservative():
    from data.download_corpus import common_corpus_license_allowed

    assert common_corpus_license_allowed("Public Domain")
    assert common_corpus_license_allowed("CC BY 4.0")
    assert common_corpus_license_allowed("Creative Commons Attribution 3.0")
    assert common_corpus_license_allowed("MIT License")
    assert common_corpus_license_allowed("Apache License 2.0")
    assert common_corpus_license_allowed("ODC-By 1.0")
    assert not common_corpus_license_allowed("CC BY-NC 4.0")
    assert not common_corpus_license_allowed("CC BY-SA 4.0")
    assert not common_corpus_license_allowed("CC BY-ND 4.0")
    assert not common_corpus_license_allowed("Various open data")
    assert not common_corpus_license_allowed(None)


def test_corpus_mix_round_robins_sources_and_preserves_row_provenance(tmp_path):
    from data.download_corpus import _write_document, mix_corpus_files

    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first_manifest = tmp_path / "first.manifest.jsonl"
    second_manifest = tmp_path / "second.manifest.jsonl"

    with first.open("w", encoding="utf-8") as stream:
        _write_document(stream, "First source document one.")
        _write_document(stream, "First source document two.")
    with second.open("w", encoding="utf-8") as stream:
        _write_document(stream, "Second source document one.")
        _write_document(stream, "Second source document two.")

    first_manifest.write_text(
        '{"dataset":"common_corpus","license":"CC BY 4.0","identifier":"doc-a"}\n'
        '{"dataset":"common_corpus","license":"Public Domain","identifier":"doc-b"}\n',
        encoding="utf-8",
    )
    second_manifest.write_text(
        '{"dataset":"cosmopedia","license":"Apache-2.0","subset":"stories"}\n'
        '{"dataset":"cosmopedia","license":"Apache-2.0","subset":"openstax"}\n',
        encoding="utf-8",
    )

    output = tmp_path / "mixed.txt"
    provenance = tmp_path / "mixed.provenance.jsonl"
    counts = mix_corpus_files(output, provenance, [
        (first, first_manifest, {"dataset": "common_corpus"}),
        (second, second_manifest, {"dataset": "cosmopedia"}),
    ])

    content = output.read_text(encoding="utf-8")
    assert content.index("First source document one.") < content.index("Second source document one.")
    assert content.index("Second source document one.") < content.index("First source document two.")
    records = [json.loads(line) for line in provenance.read_text(encoding="utf-8").splitlines()]
    assert [item["identifier"] for item in records if item["dataset"] == "common_corpus"] == [
        "doc-a", "doc-b"
    ]
    assert records[1]["license"] == "Apache-2.0"
    assert counts == {"common_corpus": 2, "cosmopedia": 2}
