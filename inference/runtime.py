"""On-demand Backblaze B2 loader and lightweight NumPy inference runtime.

The model stays in B2. A verified inference bundle is downloaded to /tmp only
when the first request needs it, then evicted from memory and disk after an idle
period. Serverless instances can be frozen/recycled at any point, so every cold
load is able to retrieve the active model again.
"""
from __future__ import annotations

import atexit
import gc
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
    "You are CatAI, an honest and helpful AI assistant. Preserve the user's intent, "
    "follow instructions, distinguish facts from guesses, and never claim you performed "
    "an action unless it actually succeeded. If you are uncertain, say so plainly."
)

_LOCK = threading.RLock()
_MODEL: dict[str, Any] | None = None
_LAST_USED = 0.0
_EVICT_TIMER: threading.Timer | None = None


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
    if _MODEL is not None and now - _LAST_USED >= IDLE_SECONDS:
        _evict_locked()


def _scheduled_evict(expected_last_used: float) -> None:
    global _MODEL, _EVICT_TIMER
    with _LOCK:
        if _MODEL is not None and _LAST_USED == expected_last_used and time.monotonic() - _LAST_USED >= IDLE_SECONDS:
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

    if token_type == "char":
        def encode(text: str) -> list[int]:
            if unk is None:
                unknown = [ch for ch in text if ch not in lookup]
                if unknown:
                    raise ValueError("Prompt contains unsupported characters")
            return [lookup.get(ch, unk) for ch in text]
        return encode, lambda ids: "".join(
            "" if token == eos else ("�" if token == unk else str(vocabulary[token]))
            for token in ids
        ).split("\0", 1)[0], eos

    if token_type == "bpe":
        merges = tokenizer.get("merges", [])
        normalized_merges = [(str(pair[0]), str(pair[1])) for pair in merges]
        def encode(text: str) -> list[int]:
            symbols = list(text)
            for left, right in normalized_merges:
                rewritten = []
                index = 0
                while index < len(symbols):
                    if index + 1 < len(symbols) and symbols[index] == left and symbols[index + 1] == right:
                        rewritten.append(left + right)
                        index += 2
                    else:
                        rewritten.append(symbols[index])
                        index += 1
                symbols = rewritten
            return [lookup.get(symbol, unk) for symbol in symbols]
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


def _sample(logits: np.ndarray, temperature: float, top_k: int, rng: np.random.Generator) -> int:
    logits = logits.astype(np.float64, copy=False) / temperature
    if top_k > 0 and top_k < logits.size:
        kept = np.argpartition(logits, -top_k)[-top_k:]
        mask = np.ones(logits.shape, dtype=bool)
        mask[kept] = False
        logits = logits.copy()
        logits[mask] = -np.inf
    logits -= np.max(logits)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum()
    return int(rng.choice(logits.size, p=probabilities))


def generate_stream(
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = 0.8,
    top_k: int = 40,
    chunk_tokens: int = 4,
) -> Iterator[dict[str, Any]]:
    """Yield small text deltas followed by final generation metadata."""
    global _LAST_USED, _EVICT_TIMER
    if not isinstance(messages, list) or not messages or len(messages) > MAX_PROMPT_MESSAGES:
        raise ValueError(f"messages must contain between 1 and {MAX_PROMPT_MESSAGES} entries")
    clean_messages = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in ("system", "user", "assistant", "file"):
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

    with _LOCK:
        _evict_if_idle_locked(time.monotonic())
        loaded = _load_model_locked()
        _touch_model_locked()
        metadata = loaded["metadata"]
        model_meta = metadata["model"]
        context_length = int(model_meta["max_seq_len"])
        encode, decode, eos = _tokenizer_from_metadata(metadata)
        if not any(message["role"] == "system" for message in clean_messages):
            clean_messages.insert(0, {"role": "system", "content": DEFAULT_SYSTEM_PROMPT})
        prompt = "".join(f"<|{item['role']}|>\n{item['content']}\n" for item in clean_messages)
        prompt += "<|assistant|>\n"
        input_tokens = encode(prompt)[-context_length:]
        if not input_tokens:
            raise ValueError("The prompt produced no model tokens")

        rng = np.random.default_rng()
        generated: list[int] = []
        sent_chars = 0
        for _ in range(int(max_new_tokens)):
            context = (input_tokens + generated)[-context_length:]
            logits = _forward(loaded["weights"], model_meta, context)
            next_token = _sample(logits, float(temperature), int(top_k), rng)
            if eos is not None and next_token == eos:
                break
            generated.append(next_token)
            if len(generated) % int(chunk_tokens) == 0:
                current_text = decode(generated)
                delta = current_text[sent_chars:]
                if delta:
                    yield {"event": "delta", "text": delta}
                    sent_chars = len(current_text)

        text = decode(generated)
        delta = text[sent_chars:]
        if delta:
            yield {"event": "delta", "text": delta}
        _touch_model_locked()
        yield {
            "event": "done",
            "text": text,
            "model": "purr-1.0",
            "model_revision": loaded["sha256"][:12],
            "tokens_in": len(input_tokens),
            "tokens_out": len(generated),
            "context_limit": context_length,
        }


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
