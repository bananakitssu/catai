from __future__ import annotations

import json

import pytest
import torch

from catai.shards import TokenShardDataset, TokenShardWriter, read_token_shards


def test_writer_splits_tokens_and_writes_manifest(tmp_path):
    manifest = TokenShardWriter(tmp_path, tokens_per_shard=3).write(range(8))

    assert manifest["total_tokens"] == 8
    assert [shard["tokens"] for shard in manifest["shards"]] == [3, 3, 2]
    assert read_token_shards(tmp_path) == list(range(8))
    assert json.loads((tmp_path / "manifest.json").read_text()) == manifest


def test_dataset_streams_overlapping_windows(tmp_path):
    TokenShardWriter(tmp_path, tokens_per_shard=3).write(range(6))

    dataset = TokenShardDataset(tmp_path, sequence_length=3)
    windows = list(dataset)

    expected = [
        torch.tensor([0, 1, 2]),
        torch.tensor([1, 2, 3]),
        torch.tensor([2, 3, 4]),
        torch.tensor([3, 4, 5]),
    ]
    assert len(windows) == len(expected)
    assert all(torch.equal(actual, wanted) for actual, wanted in zip(windows, expected))
    assert len(dataset) == 4


def test_writer_rejects_empty_stream(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        TokenShardWriter(tmp_path).write([])


def test_dataset_rejects_invalid_manifest(tmp_path):
    (tmp_path / "manifest.json").write_text('{"format": 999, "dtype": "uint32", "shards": []}')

    with pytest.raises(ValueError, match="unsupported"):
        TokenShardDataset(tmp_path)


def test_dataset_detects_truncated_shard(tmp_path):
    TokenShardWriter(tmp_path, tokens_per_shard=4).write(range(4))
    (tmp_path / "shard_00000.bin").write_bytes(b"\x00\x00")

    with pytest.raises(ValueError, match="size does not match"):
        list(TokenShardDataset(tmp_path, sequence_length=2))


def test_writer_rejects_token_outside_uint32(tmp_path):
    with pytest.raises(ValueError, match="uint32"):
        TokenShardWriter(tmp_path).write([0, 2**32])
