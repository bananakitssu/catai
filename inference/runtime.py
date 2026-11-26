"""On-demand Backblaze B2 loader and lightweight NumPy inference runtime.

The model stays in B2. A verified inference bundle is downloaded to /tmp only
when the first request needs it, then evicted from memory and disk after an idle
period. Serverless instances can be frozen/recycled at any point, so every cold
load is able to retrieve the active model again.
"""
from __future__ import annotations

import atexit
import gc
import heapq
import hashlib
import logging
import json
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

import boto3
import numpy as np
from botocore.config import Config

MODEL_FILE = Path(tempfile.gettempdir()) / "catai-sft.npz"
IDLE_SECONDS = max(15, int(os.environ.get("CATAI_MODEL_IDLE_SECONDS", "60")))
MAX_PROMPT_MESSAGES = 40
DEFAULT_MAX_NEW_TOKENS = 160
MAX_NEW_TOKENS = 256
DEFAULT_SYSTEM_PROMPT = (
    "You are CatAI, a friendly cat-themed conversational AI. "
    "Be helpful, honest, and concise. Use :3 occasionally, not in every response.\n"
    "CatAI internal state:\n"
    '{"emotions":{"happiness":0.5,"sadness":0.1,"affection":0.5,"curiosity":0.7,'
    '"excitement":0.3,"frustration":0.1,"anger":0.0,"fear":0.1,"calmness":0.6,'
    '"confidence":0.7,"loneliness":0.1,"playfulness":0.4},'
    '"needs":{"social_connection":0.5,"stimulation":0.5,"task_completion":0.6,"rest":0.5},'
    '"personality":{"playful":0.9,"curious":0.85,"helpful":0.8,"affectionate":0.9,"seriousness":0.3}}'
)

SUPPORTED_MESSAGE_ROLES = frozenset({
    "system",
    "user",
    "assistant",
    "file",
    "moderator",
    "tool_call:assistant",
    "tool_call:result->assistant",
    "system-1",
    "mcp_tool_call:assistant",
    "mcp_tool_call:result->assistant",
    "error",
    "tool_call:error",
    "mcp_tool_call:error",
    "assistant:Gemini",
    "assistant:Claude",
    "assistant:ChatGPT",
    "assistant:Qwen",
    "assistant:DeepSeek",
})
ASSISTANT_OUTPUT_ROLES = frozenset({
    "assistant",
    "tool_call:assistant",
    "mcp_tool_call:assistant",
    "assistant:Gemini",
    "assistant:Claude",
    "assistant:ChatGPT",
    "assistant:Qwen",
    "assistant:DeepSeek",
})


def _ensure_default_system_message(
    messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Prepend CatAI's trained state prompt and retain leading custom system text."""
    remaining = list(messages)
    supplied_system: list[str] = []
    while remaining and remaining[0]["role"] == "system":
        supplied_system.append(remaining.pop(0)["content"])

    system_content = DEFAULT_SYSTEM_PROMPT
    if supplied_system:
        if supplied_system[0] == DEFAULT_SYSTEM_PROMPT:
            supplied_system = supplied_system[1:]
        if supplied_system:
            system_content += "\n" + "\n".join(supplied_system)

    return [{"role": "system", "content": system_content}, *remaining]


def _serialize_messages(messages: list[dict[str, str]]) -> str:
    """Serialize history exactly like SFT, including assistant EOS targets."""
    parts: list[str] = []
    for message in messages:
        role = message["role"]
        parts.append(f"<|{role}|>\n{message['content']}\n")
        if role in ASSISTANT_OUTPUT_ROLES:
            parts.append("<EOS>")
    parts.append("<|assistant|>\n")
    return "".join(parts)

def _fit_messages_to_context(
    messages: list[dict[str, str]],
    encode,
    context_length: int,
) -> tuple[list[dict[str, str]], list[int]]:
    """Keep the system prompt and newest whole messages before truncating text."""
    if not messages:
        return [], []
    system = messages[0]
    history = messages[1:]
    selected: list[dict[str, str]] = []
    base_tokens = encode(_serialize_messages([system]))
    if len(base_tokens) >= context_length:
        return [system], base_tokens[-context_length:]

    for message in reversed(history):
        candidate = [system, message, *reversed(selected)]
        candidate_tokens = encode(_serialize_messages(candidate))
        if len(candidate_tokens) <= context_length:
            selected.append(message)
            continue

        # The newest message gets priority even when it alone is too long.
        if not selected:
            low, high = 0, len(message["content"])
            best: list[int] = []
            best_length = 0
            while low <= high:
                middle = (low + high) // 2
                shortened = {**message, "content": message["content"][-middle:] if middle else ""}
                trial = encode(_serialize_messages([system, shortened]))
                if len(trial) <= context_length:
                    best = trial
                    best_length = middle
                    low = middle + 1
                else:
                    high = middle - 1
            if best:
                return [system, {**message, "content": message["content"][-best_length:] if best_length else ""}], best
        break

    fitted = [system, *reversed(selected)]
    return fitted, encode(_serialize_messages(fitted))

_LOCK = threading.RLock()
_MODEL: dict[str, Any] | None = None
_LAST_USED = 0.0
_EVICT_TIMER: threading.Timer | None = None
_ACTIVE_GENERATIONS = 0


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is not configured")
    return value


def _b2_client():
    # Accept either a complete HTTPS URL or the bare S3 hostname copied from
    # the Backblaze dashboard/GitHub Actions secret. Normalize both the same
    # way the publisher does, including accidental surrounding quote marks.
    endpoint = _required_env("B2_ENDPOINT_URL").strip().strip(chr(34) + chr(39)).strip()
    if "://" not in endpoint:
        endpoint = "https://" + endpoint
    parsed = urlparse(endpoint)
    if (
        parsed.scheme.lower() != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in ("", "/")
    ):
        raise RuntimeError(
            "B2_ENDPOINT_URL must be an HTTPS Backblaze S3 URL or a bare "
            "s3.<region>.backblazeb2.com hostname"
        )
    region_match = re.fullmatch(r"s3\.([a-z0-9-]+)\.backblazeb2\.com", parsed.hostname.lower())
    if region_match is None:
        raise RuntimeError("B2_ENDPOINT_URL must use the exact s3.<region>.backblazeb2.com hostname")
    endpoint_region = region_match.group(1)
    region = os.environ.get("B2_REGION", "").strip() or endpoint_region
    if region != endpoint_region:
        raise RuntimeError("B2_REGION must match the region in B2_ENDPOINT_URL")
    # Use a clean API root with no copied trailing slash.
    endpoint_url = f"https://{parsed.netloc}"
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        region_name=region,
        aws_access_key_id=_required_env("B2_KEY_ID"),
        aws_secret_access_key=_required_env("B2_APPLICATION_KEY"),
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _download_active_bundle() -> dict[str, Any]:
    bucket = _required_env("B2_BUCKET_NAME")
    manifest_key = os.environ.get("CATAI_MODEL_KEY", "ai/models/current.json").strip()
    if not manifest_key or manifest_key.startswith("/") or ".." in manifest_key.split("/"):
        raise RuntimeError("CATAI_MODEL_KEY is invalid")
    client = _b2_client()

    response = client.get_object(Bucket=bucket, Key=manifest_key)
    try:
        raw_manifest = response["Body"].read(64 * 1024 + 1)
    finally:
        response["Body"].close()
    if len(raw_manifest) > 64 * 1024:
        raise RuntimeError("Active model manifest is too large")
    manifest = json.loads(raw_manifest.decode("utf-8"))
    object_key = manifest.get("object_key")
    expected_hash = manifest.get("sha256")
    expected_size = manifest.get("bytes")
    if (
        not isinstance(object_key, str)
        or not object_key.startswith("ai/models/releases/")
        or not object_key.endswith(".npz")
        or ".." in object_key.split("/")
        or not isinstance(expected_hash, str)
        or not re.fullmatch(r"[0-9a-f]{64}", expected_hash)
        or not isinstance(expected_size, int)
        or expected_size <= 0
    ):
        raise RuntimeError("Active model manifest is invalid")

    partial = MODEL_FILE.with_suffix(".npz.part")
    partial.unlink(missing_ok=True)
    try:
        client.download_file(bucket, object_key, str(partial))
        actual_hash, actual_size = _hash_file(partial)
        if actual_size != expected_size or actual_hash != expected_hash:
            raise RuntimeError("Downloaded model failed size/SHA-256 verification")
        with np.load(partial, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["__metadata__"].item()))
            if metadata.get("format") != "catai-numpy-inference" or metadata.get("format_version") != 1:
                raise RuntimeError("Unsupported CatAI inference bundle")
            model_meta = metadata.get("model")
            tokenizer_meta = metadata.get("tokenizer")
            if not isinstance(model_meta, dict) or not isinstance(tokenizer_meta, dict):
                raise RuntimeError("Inference bundle metadata is incomplete")
            if not all(key in model_meta for key in ("max_seq_len", "d_model", "n_heads", "n_layers")):
                raise RuntimeError("Inference bundle model dimensions are incomplete")
            weights = {
                name: np.asarray(archive[name], dtype=np.float32)
                for name in archive.files
                if name != "__metadata__"
            }
        # CatAI ties input and output embeddings, so the export may omit the
        # duplicate output embedding to save bundle space.
        if "lm_head.weight" not in weights and "token_embedding.weight" in weights:
            weights["lm_head.weight"] = weights["token_embedding.weight"]
        _validate_shapes(weights, model_meta, tokenizer_meta)
        os.replace(partial, MODEL_FILE)
        logging.getLogger(__name__).info(
            "Loaded CatAI inference bundle revision=%s bytes=%d",
            actual_hash[:12],
            actual_size,
        )
        return {
            "weights": weights,
            "metadata": metadata,
            "object_key": object_key,
            "sha256": actual_hash,
        }
    except Exception:
        partial.unlink(missing_ok=True)
        MODEL_FILE.unlink(missing_ok=True)
        raise


def _validate_shapes(weights: dict[str, np.ndarray], model: dict[str, Any], tokenizer: dict[str, Any]) -> None:
    vocab = tokenizer.get("vocabulary")
    d_model = int(model["d_model"])
    layers = int(model["n_layers"])
    heads = int(model["n_heads"])
    if not isinstance(vocab, list) or not vocab or d_model < 1 or heads < 1 or d_model % heads:
        raise RuntimeError("Inference model dimensions or vocabulary are invalid")
    if weights.get("token_embedding.weight", np.empty((0, 0))).shape != (len(vocab), d_model):
        raise RuntimeError("Token embedding shape does not match metadata")
    required = ["norm.weight", "lm_head.weight"]
    for index in range(layers):
        prefix = f"blocks.{index}."
        required.extend([
            prefix + "norm1.weight",
            prefix + "attention.qkv.weight",
            prefix + "attention.out_proj.weight",
            prefix + "norm2.weight",
            prefix + "mlp.w1.weight",
            prefix + "mlp.w2.weight",
            prefix + "mlp.w3.weight",
        ])
    missing = [name for name in required if name not in weights]
    if missing:
        raise RuntimeError("Inference model is missing required weights: " + ", ".join(missing[:5]))
    if weights["lm_head.weight"].shape != (len(vocab), d_model):
        raise RuntimeError("LM head shape does not match metadata")


def _evict_locked() -> None:
    global _MODEL, _EVICT_TIMER
    _MODEL = None
    if _EVICT_TIMER is not None:
        _EVICT_TIMER.cancel()
        _EVICT_TIMER = None
    MODEL_FILE.unlink(missing_ok=True)
    MODEL_FILE.with_suffix(".npz.part").unlink(missing_ok=True)
    gc.collect()


def _evict_if_idle_locked(now: float) -> None:
    if _MODEL is not None and _ACTIVE_GENERATIONS == 0 and now - _LAST_USED >= IDLE_SECONDS:
        _evict_locked()


def _scheduled_evict(expected_last_used: float) -> None:
    global _MODEL, _EVICT_TIMER
    with _LOCK:
        if _MODEL is not None and _ACTIVE_GENERATIONS == 0 and _LAST_USED == expected_last_used and time.monotonic() - _LAST_USED >= IDLE_SECONDS:
            _evict_locked()
        _EVICT_TIMER = None


def _touch_model_locked() -> None:
    global _LAST_USED, _EVICT_TIMER
    _LAST_USED = time.monotonic()
    if _EVICT_TIMER is not None:
        _EVICT_TIMER.cancel()
    _EVICT_TIMER = threading.Timer(IDLE_SECONDS, _scheduled_evict, args=(_LAST_USED,))
    _EVICT_TIMER.daemon = True
    _EVICT_TIMER.start()


def _load_model_locked() -> dict[str, Any]:
    global _MODEL
    if _MODEL is None:
        _MODEL = _download_active_bundle()
    return _MODEL


def _tokenizer_from_metadata(metadata: dict[str, Any]):
    tokenizer = metadata["tokenizer"]
    vocabulary = tokenizer["vocabulary"]
    token_type = tokenizer.get("type")
    lookup = {token: index for index, token in enumerate(vocabulary)}
    unk = lookup.get("<UNK>")
    eos = lookup.get("<EOS>")

    def special_eos_parts(text: str) -> list[str]:
        return text.split("<EOS>")

    def encode_char(text: str) -> list[int]:
        result: list[int] = []
        parts = special_eos_parts(text)
        for index, part in enumerate(parts):
            for character in part:
                token_id = lookup.get(character)
                if token_id is None:
                    if unk is None:
                        raise ValueError("Prompt contains unsupported characters")
                    token_id = unk
                result.append(token_id)
            if index < len(parts) - 1:
                if eos is None:
                    raise ValueError("Checkpoint tokenizer has no EOS token")
                result.append(eos)
        return result

    def decode_char(ids: list[int]) -> str:
        pieces: list[str] = []
        for token in ids:
            if token == eos:
                break
            if token == unk:
                pieces.append("�")
            else:
                pieces.append(str(vocabulary[token]))
        return "".join(pieces)

    if token_type == "char":
        return encode_char, decode_char, eos

    if token_type == "bpe":
        merges = tokenizer.get("merges", [])
        normalized_merges = [(str(pair[0]), str(pair[1])) for pair in merges]

        merge_rank = {pair: rank for rank, pair in enumerate(normalized_merges)}

        def encode_piece(text: str) -> list[int]:
            # Apply the same lowest-rank-first BPE merges as training, but only
            # revisit adjacent pairs affected by a merge (rather than scanning
            # every merge over the entire remaining string).
            symbols = list(text)
            if len(symbols) < 2 or not merge_rank:
                merged_symbols = symbols
            else:
                count = len(symbols)
                previous = [index - 1 for index in range(count)]
                following = [index + 1 if index + 1 < count else -1 for index in range(count)]
                alive = [True] * count
                heap: list[tuple[int, int, int]] = []

                def push_pair(left_index: int) -> None:
                    right_index = following[left_index]
                    if right_index < 0 or not alive[left_index] or not alive[right_index]:
                        return
                    rank = merge_rank.get((symbols[left_index], symbols[right_index]))
                    if rank is not None:
                        heapq.heappush(heap, (rank, left_index, right_index))

                for index in range(count - 1):
                    push_pair(index)

                while heap:
                    rank, left_index, right_index = heapq.heappop(heap)
                    if (
                        not alive[left_index]
                        or not alive[right_index]
                        or following[left_index] != right_index
                        or merge_rank.get((symbols[left_index], symbols[right_index])) != rank
                    ):
                        continue
                    symbols[left_index] += symbols[right_index]
                    alive[right_index] = False
                    next_index = following[right_index]
                    following[left_index] = next_index
                    if next_index >= 0:
                        previous[next_index] = left_index
                    push_pair(previous[left_index]) if previous[left_index] >= 0 else None
                    push_pair(left_index)

                merged_symbols = []
                index = 0
                while index >= 0:
                    if alive[index]:
                        merged_symbols.append(symbols[index])
                    index = following[index]

            token_ids = []
            for symbol in merged_symbols:
                token_id = lookup.get(symbol)
                if token_id is None:
                    if unk is None:
                        raise ValueError("Prompt contains unsupported text pieces")
                    token_id = unk
                token_ids.append(token_id)
            return token_ids

        def encode(text: str) -> list[int]:
            result: list[int] = []
            parts = special_eos_parts(text)
            for index, part in enumerate(parts):
                result.extend(encode_piece(part))
                if index < len(parts) - 1:
                    if eos is None:
                        raise ValueError("Checkpoint tokenizer has no EOS token")
                    result.append(eos)
            return result

        def decode(ids: list[int]) -> str:
            pieces = []
            for token in ids:
                if token == eos:
                    break
                pieces.append("�" if token == unk else str(vocabulary[token]))
            return "".join(pieces)

        return encode, decode, eos
    raise RuntimeError("Unsupported CatAI tokenizer type")


def _rms_norm(values: np.ndarray, scale: np.ndarray) -> np.ndarray:
    variance = np.mean(values * values, axis=-1, keepdims=True)
    return (values / np.sqrt(variance + 1e-6)) * scale


def _rotate_half(values: np.ndarray) -> np.ndarray:
    half = values.shape[-1] // 2
    return np.concatenate((-values[..., half:], values[..., :half]), axis=-1)


def _forward(weights: dict[str, np.ndarray], model: dict[str, Any], tokens: list[int]) -> np.ndarray:
    d_model = int(model["d_model"])
    n_heads = int(model["n_heads"])
    n_layers = int(model["n_layers"])
    head_dim = d_model // n_heads
    x = weights["token_embedding.weight"][np.asarray(tokens, dtype=np.int64)]
    seq_len = x.shape[0]

    inv_freq = 1.0 / (10000.0 ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    freqs = np.outer(np.arange(seq_len, dtype=np.float32), inv_freq)
    angles = np.concatenate((freqs, freqs), axis=-1)
    cos = np.cos(angles)[None, :, :].astype(np.float32, copy=False)
    sin = np.sin(angles)[None, :, :].astype(np.float32, copy=False)
    causal_mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)[None, :, :]

    for index in range(n_layers):
        prefix = f"blocks.{index}."
        norm_x = _rms_norm(x, weights[prefix + "norm1.weight"])
        qkv = norm_x @ weights[prefix + "attention.qkv.weight"].T
        qkv = qkv.reshape(seq_len, 3, n_heads, head_dim).transpose(1, 2, 0, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = (q * cos) + (_rotate_half(q) * sin)
        k = (k * cos) + (_rotate_half(k) * sin)
        scores = (q @ k.transpose(0, 2, 1)) * (head_dim ** -0.5)
        scores = np.where(causal_mask, -1.0e30, scores)
        scores -= np.max(scores, axis=-1, keepdims=True)
        attention = np.exp(scores)
        attention /= np.sum(attention, axis=-1, keepdims=True)
        attended = attention @ v
        attended = attended.transpose(1, 0, 2).reshape(seq_len, d_model)
        x = x + attended @ weights[prefix + "attention.out_proj.weight"].T

        norm_x = _rms_norm(x, weights[prefix + "norm2.weight"])
        gate = norm_x @ weights[prefix + "mlp.w1.weight"].T
        gate = gate / (1.0 + np.exp(-np.clip(gate, -80.0, 80.0)))
        value = norm_x @ weights[prefix + "mlp.w3.weight"].T
        x = x + (gate * value) @ weights[prefix + "mlp.w2.weight"].T

    x = _rms_norm(x, weights["norm.weight"])
    return x[-1] @ weights["lm_head.weight"].T


def _forward_cached(
    weights: dict[str, np.ndarray],
    model: dict[str, Any],
    tokens: list[int],
    cache: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run a prompt or incremental token using preallocated per-layer KV buffers."""
    if not tokens:
        raise ValueError("cached forward requires at least one token")
    d_model = int(model["d_model"])
    n_heads = int(model["n_heads"])
    n_layers = int(model["n_layers"])
    head_dim = d_model // n_heads
    max_seq_len = int(model["max_seq_len"])
    if cache is None:
        cache = {
            "position": 0,
            "max_seq_len": max_seq_len,
            "layers": [None] * n_layers,
        }
    past_len = int(cache["position"])
    seq_len = len(tokens)
    end_position = past_len + seq_len
    if end_position > max_seq_len:
        raise ValueError("KV cache capacity exceeded; rebuild the cache for the active context window")
    x = weights["token_embedding.weight"][np.asarray(tokens, dtype=np.int64)]

    inv_freq = 1.0 / (10000.0 ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    positions = np.arange(past_len, end_position, dtype=np.float32)
    freqs = np.outer(positions, inv_freq)
    angles = np.concatenate((freqs, freqs), axis=-1)
    cos = np.cos(angles)[None, :, :].astype(np.float32, copy=False)
    sin = np.sin(angles)[None, :, :].astype(np.float32, copy=False)
    causal_mask = (
        np.arange(end_position)[None, :] > (past_len + np.arange(seq_len)[:, None])
    )

    for index in range(n_layers):
        prefix = f"blocks.{index}."
        norm_x = _rms_norm(x, weights[prefix + "norm1.weight"])
        qkv = norm_x @ weights[prefix + "attention.qkv.weight"].T
        qkv = qkv.reshape(seq_len, 3, n_heads, head_dim).transpose(1, 2, 0, 3)
        q, current_k, current_v = qkv[0], qkv[1], qkv[2]
        q = (q * cos) + (_rotate_half(q) * sin)
        current_k = (current_k * cos) + (_rotate_half(current_k) * sin)

        layer_cache = cache["layers"][index]
        if layer_cache is None:
            keys = np.empty((n_heads, max_seq_len, head_dim), dtype=np.float32)
            values = np.empty_like(keys)
            cache["layers"][index] = (keys, values)
        else:
            keys, values = layer_cache
        keys[:, past_len:end_position, :] = current_k
        values[:, past_len:end_position, :] = current_v
        active_keys = keys[:, :end_position, :]
        active_values = values[:, :end_position, :]

        scores = (q @ active_keys.transpose(0, 2, 1)) * (head_dim ** -0.5)
        scores = np.where(causal_mask[None, :, :], -1.0e30, scores)
        scores -= np.max(scores, axis=-1, keepdims=True)
        attention = np.exp(scores)
        attention /= np.sum(attention, axis=-1, keepdims=True)
        attended = attention @ active_values
        attended = attended.transpose(1, 0, 2).reshape(seq_len, d_model)
        x = x + attended @ weights[prefix + "attention.out_proj.weight"].T

        norm_x = _rms_norm(x, weights[prefix + "norm2.weight"])
        gate = norm_x @ weights[prefix + "mlp.w1.weight"].T
        gate = gate / (1.0 + np.exp(-np.clip(gate, -80.0, 80.0)))
        value = norm_x @ weights[prefix + "mlp.w3.weight"].T
        x = x + (gate * value) @ weights[prefix + "mlp.w2.weight"].T

    cache["position"] = end_position
    x = _rms_norm(x, weights["norm.weight"])
    return x[-1] @ weights["lm_head.weight"].T, cache


def _sample(logits: np.ndarray, temperature: float, top_k: int, rng: np.random.Generator) -> int:
    # Restrict softmax and sampling to the retained candidates. This avoids
    # allocating a vocabulary-sized probability vector for every generated token.
    candidate_count = min(max(1, int(top_k)), int(logits.size))
    if candidate_count < logits.size:
        candidate_ids = np.argpartition(logits, -candidate_count)[-candidate_count:]
        candidate_logits = logits[candidate_ids].astype(np.float64, copy=False) / temperature
    else:
        candidate_ids = np.arange(logits.size)
        candidate_logits = logits.astype(np.float64, copy=False) / temperature
    candidate_logits = candidate_logits - np.max(candidate_logits)
    probabilities = np.exp(candidate_logits)
    probabilities /= probabilities.sum()
    return int(candidate_ids[int(rng.choice(candidate_count, p=probabilities))])


def generate_stream(
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = 0.8,
    top_k: int = 40,
    chunk_tokens: int = 4,
) -> Iterator[dict[str, Any]]:
    """Yield incremental text and timing metadata without serializing generations."""
    global _ACTIVE_GENERATIONS
    if not isinstance(messages, list) or not messages or len(messages) > MAX_PROMPT_MESSAGES:
        raise ValueError(f"messages must contain between 1 and {MAX_PROMPT_MESSAGES} entries")
    clean_messages = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in SUPPORTED_MESSAGE_ROLES:
            raise ValueError("each message must have a supported role")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip() or len(content) > 12000:
            raise ValueError("message content must be non-empty and at most 12000 characters")
        clean_messages.append({"role": message["role"], "content": content})
    if not 1 <= int(max_new_tokens) <= MAX_NEW_TOKENS:
        raise ValueError(f"max_new_tokens must be between 1 and {MAX_NEW_TOKENS}")
    if not 0.05 <= float(temperature) <= 2.0:
        raise ValueError("temperature must be between 0.05 and 2.0")
    if not 1 <= int(top_k) <= 256:
        raise ValueError("top_k must be between 1 and 256")
    if not 1 <= int(chunk_tokens) <= 16:
        raise ValueError("chunk_tokens must be between 1 and 16")

    total_started = time.perf_counter()
    with _LOCK:
        _evict_if_idle_locked(time.monotonic())
        needs_download = _MODEL is None
    if needs_download:
        yield {"event": "status", "status": "loading_model"}

    load_started = time.perf_counter()
    with _LOCK:
        _evict_if_idle_locked(time.monotonic())
        loaded = _load_model_locked()
        model_load_ms = (time.perf_counter() - load_started) * 1000.0
        _ACTIVE_GENERATIONS += 1
        _touch_model_locked()

    try:
        metadata = loaded["metadata"]
        model_meta = metadata["model"]
        context_length = int(model_meta["max_seq_len"])
        tokenize_started = time.perf_counter()
        encode, decode, eos = _tokenizer_from_metadata(metadata)
        clean_messages = _ensure_default_system_message(clean_messages)
        clean_messages, input_tokens = _fit_messages_to_context(
            clean_messages, encode, context_length
        )
        tokenize_ms = (time.perf_counter() - tokenize_started) * 1000.0
        if not input_tokens:
            raise ValueError("The prompt produced no model tokens")

        yield {"event": "status", "status": "connecting"}
        rng = np.random.default_rng()
        generated: list[int] = []
        sent_tokens = 0
        leading_whitespace = ""
        has_visible_text = False
        cache = None
        logits = None
        prefill_ms = 0.0
        decode_compute_seconds = 0.0
        first_token_ms = None
        flush_count = 0
        for token_index in range(int(max_new_tokens)):
            if cache is None:
                context = (input_tokens + generated)[-context_length:]
                forward_started = time.perf_counter()
                logits, cache = _forward_cached(loaded["weights"], model_meta, context)
                forward_elapsed = time.perf_counter() - forward_started
                if token_index == 0:
                    prefill_ms = forward_elapsed * 1000.0
                else:
                    decode_compute_seconds += forward_elapsed
            sample_started = time.perf_counter()
            next_token = _sample(logits, float(temperature), int(top_k), rng)
            decode_compute_seconds += time.perf_counter() - sample_started
            if first_token_ms is None:
                first_token_ms = (time.perf_counter() - total_started) * 1000.0
            if eos is not None and next_token == eos:
                break
            generated.append(next_token)
            pending_count = len(generated) - sent_tokens
            if pending_count:
                new_text = decode(generated[sent_tokens:])
                should_flush = (
                    pending_count >= int(chunk_tokens)
                    or (not has_visible_text and bool(new_text.strip()))
                    or len(new_text) >= 40
                    or new_text.endswith((".", "!", "?", "\n", ":", ";"))
                )
                if should_flush:
                    sent_tokens = len(generated)
                    delta = new_text
                    if not delta:
                        continue
                    if not has_visible_text:
                        leading_whitespace += delta
                        if not leading_whitespace.strip():
                            continue
                        delta = leading_whitespace
                        leading_whitespace = ""
                        has_visible_text = True
                    yield {"event": "delta", "text": delta}
                    flush_count += 1

            # Deliver a visible delta before spending time computing the next
            # token's logits, reducing first-token latency for streamed clients.
            if token_index + 1 < int(max_new_tokens) and len(input_tokens) + len(generated) <= context_length:
                forward_started = time.perf_counter()
                logits, cache = _forward_cached(
                    loaded["weights"], model_meta, [next_token], cache
                )
                decode_compute_seconds += time.perf_counter() - forward_started
            else:
                # Rebase RoPE positions after the sliding context window advances.
                cache = None

        final_delta = decode(generated[sent_tokens:])
        if final_delta:
            if not has_visible_text:
                leading_whitespace += final_delta
                if leading_whitespace.strip():
                    yield {"event": "delta", "text": leading_whitespace}
                    flush_count += 1
            else:
                yield {"event": "delta", "text": final_delta}
                flush_count += 1
        decode_ms = decode_compute_seconds * 1000.0
        generation_ms = (time.perf_counter() - total_started) * 1000.0
        with _LOCK:
            _touch_model_locked()
        yield {
            "event": "done",
            "text": decode(generated),
            "model": "purr-1.0",
            "model_revision": loaded["sha256"][:12],
            "tokens_in": len(input_tokens),
            "tokens_out": len(generated),
            "context_limit": context_length,
            "metrics": {
                "model_load_ms": round(model_load_ms, 2),
                "tokenize_ms": round(tokenize_ms, 2),
                "prefill_ms": round(prefill_ms, 2),
                "time_to_first_token_ms": round(first_token_ms or generation_ms, 2),
                "decode_ms": round(decode_ms, 2),
                "total_ms": round(generation_ms, 2),
                "tokens_per_second": round(len(generated) / max(decode_ms / 1000.0, 1e-9), 2),
                "stream_flush_count": flush_count,
            },
        }
    finally:
        with _LOCK:
            _ACTIVE_GENERATIONS = max(0, _ACTIVE_GENERATIONS - 1)
            _touch_model_locked()


def generate(
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = 0.8,
    top_k: int = 40,
) -> dict[str, Any]:
    """Compatibility wrapper that collects a streamed generation."""
    final: dict[str, Any] | None = None
    for event in generate_stream(
        messages,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    ):
        if event.get("event") == "done":
            final = event
    if final is None:
        raise RuntimeError("CatAI generation did not finish")
    return {key: value for key, value in final.items() if key != "event"}


def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "catai-inference",
        "model_loaded": _MODEL is not None,
        "model_file_cached": MODEL_FILE.exists(),
        "idle_timeout_seconds": IDLE_SECONDS,
    }


def shutdown() -> None:
    with _LOCK:
        _evict_locked()


atexit.register(shutdown)
