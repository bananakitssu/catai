"""Disk-backed token shards for scalable CatAI pretraining."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import torch
from torch.utils.data import IterableDataset


_UINT32 = struct.Struct("<I")


class TokenShardWriter:
    """Write a token stream into fixed-size little-endian uint32 shard files."""

    def __init__(self, output_dir: str | Path, *, tokens_per_shard: int = 1_000_000) -> None:
        if tokens_per_shard < 1:
            raise ValueError("tokens_per_shard must be positive")
        self.output_dir = Path(output_dir)
        self.tokens_per_shard = tokens_per_shard

    def write(self, tokens: Iterable[int]) -> dict:
        """Write tokens and return the generated manifest."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        shard_paths: list[dict[str, int | str]] = []
        shard_index = 0
        shard_tokens = 0
        total_tokens = 0
        handle = None

        try:
            for token in tokens:
                token = int(token)
                if token < 0 or token > 0xFFFFFFFF:
                    raise ValueError("token ids must fit in uint32")
                if handle is None or shard_tokens >= self.tokens_per_shard:
                    if handle is not None:
                        handle.close()
                    filename = f"shard_{shard_index:05d}.bin"
                    handle = (self.output_dir / filename).open("wb")
                    shard_paths.append({"path": filename, "tokens": 0})
                    shard_index += 1
                    shard_tokens = 0
                handle.write(_UINT32.pack(token))
                shard_tokens += 1
                total_tokens += 1
                shard_paths[-1]["tokens"] = shard_tokens
        finally:
            if handle is not None:
                handle.close()

        if total_tokens == 0:
            raise ValueError("cannot write an empty token stream")

        manifest = {
            "format": 1,
            "dtype": "uint32",
            "tokens_per_shard": self.tokens_per_shard,
            "total_tokens": total_tokens,
            "shards": shard_paths,
        }
        (self.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        return manifest


class TokenShardDataset(IterableDataset[torch.Tensor]):
    """Stream fixed-length token windows from a shard directory."""

    def __init__(self, shard_dir: str | Path, *, sequence_length: int = 128) -> None:
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        self.shard_dir = Path(shard_dir)
        self.manifest = self._load_manifest()
        self.sequence_length = sequence_length

    def _load_manifest(self) -> dict:
        path = self.shard_dir / "manifest.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing shard manifest: {path}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("format") != 1 or manifest.get("dtype") != "uint32":
            raise ValueError("unsupported token shard format")
        if not isinstance(manifest.get("shards"), list) or not manifest["shards"]:
            raise ValueError("manifest must contain at least one shard")
        return manifest

    def _iter_tokens(self) -> Iterator[int]:
        for shard in self.manifest["shards"]:
            path = self.shard_dir / shard["path"]
            expected = int(shard["tokens"])
            if path.stat().st_size != expected * _UINT32.size:
                raise ValueError(f"shard size does not match manifest: {path}")
            with path.open("rb") as handle:
                for _ in range(expected):
                    raw = handle.read(_UINT32.size)
                    if len(raw) != _UINT32.size:
                        raise ValueError(f"truncated token shard: {path}")
                    yield _UINT32.unpack(raw)[0]

    def __iter__(self) -> Iterator[torch.Tensor]:
        window: list[int] = []
        for token in self._iter_tokens():
            window.append(token)
            if len(window) == self.sequence_length:
                yield torch.tensor(window, dtype=torch.long)
                window.pop(0)

    def __len__(self) -> int:
        total = int(self.manifest["total_tokens"])
        return max(0, total - self.sequence_length + 1)


def read_token_shards(shard_dir: str | Path) -> Sequence[int]:
    """Read every token from a shard set; primarily useful for tests and inspection."""
    return list(TokenShardDataset(shard_dir, sequence_length=2)._iter_tokens())
