"""Generate a small, repeatable set of chat responses for manual evaluation."""

from __future__ import annotations

import argparse

import torch

from catai.checkpoint import load_checkpoint, load_checkpoint_metadata
from catai.chat_dataset import DEFAULT_STATE, state_system_message
from catai.device import resolve_device
from catai.generate_cli import _load_tokenizer
from catai.generation import generate
from catai.model import CatAI
from catai.sft import ROLE_HEADER


PROMPTS = (
    ("Greeting", "Hello!"),
    ("Programming", "What is a variable in programming?"),
    ("HTTP", "What does HTTP 404 mean?"),
    ("Recursion", "Explain recursion in one sentence."),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate CatAI on a fixed set of short chat prompts."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.65)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-unk-fraction", type=float, default=0.10, help="Fail if more than this fraction of generated tokens are UNK.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_new_tokens < 0:
        raise ValueError("--max-new-tokens must be non-negative")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if args.top_k < 1:
        raise ValueError("--top-k must be positive")
    if args.repetition_penalty < 1.0:
        raise ValueError("--repetition-penalty must be at least 1.0")
    if args.no_repeat_ngram_size < 0:
        raise ValueError("--no-repeat-ngram-size must be non-negative")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if not 0.0 <= args.max_unk_fraction <= 1.0:
        raise ValueError("--max-unk-fraction must be in [0, 1]")

    metadata = load_checkpoint_metadata(args.checkpoint)
    model_metadata = metadata.get("model")
    tokenizer_metadata = metadata.get("tokenizer")
    if not isinstance(model_metadata, dict) or not isinstance(tokenizer_metadata, dict):
        raise ValueError("checkpoint is missing model/tokenizer metadata")

    tokenizer = _load_tokenizer(tokenizer_metadata)
    model = CatAI(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=int(model_metadata["max_seq_len"]),
        d_model=int(model_metadata["d_model"]),
        n_heads=int(model_metadata["n_heads"]),
        n_layers=int(model_metadata["n_layers"]),
    )
    device = resolve_device(args.device)
    model.to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    torch.manual_seed(args.seed)
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    failed_checks: list[str] = []

    system_message = state_system_message(DEFAULT_STATE)
    for label, question in PROMPTS:
        messages = [
            system_message,
            {"role": "user", "content": question},
        ]
        prompt = "".join(
            ROLE_HEADER.format(role=message["role"])
            + message["content"]
            + "\n"
            for message in messages
        ) + ROLE_HEADER.format(role="assistant")
        prompt_tokens = tokenizer.encode(prompt)
        input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        output_ids = generate(
            model,
            input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=min(args.top_k, tokenizer.vocab_size),
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            eos_token_id=tokenizer.eos_token_id,
        )
        reply_ids = output_ids[0, len(prompt_tokens):].detach().cpu().tolist()
        reply = tokenizer.decode(reply_ids).strip()
        visible_ids = [token for token in reply_ids if token != tokenizer.eos_token_id]
        unk_fraction = (
            sum(token == unk_token_id for token in visible_ids) / len(visible_ids)
            if visible_ids else 0.0
        )
        print(f"\n=== {label} ===")
        print(f"User: {question}")
        print(f"CatAI: {reply or '[empty response]'}")
        print(f"Generation tokens: {len(visible_ids)}; UNK fraction: {unk_fraction:.2%}", flush=True)
        if not reply:
            failed_checks.append(f"{label}: empty response")
        if unk_fraction > args.max_unk_fraction:
            failed_checks.append(f"{label}: UNK fraction {unk_fraction:.2%} exceeds {args.max_unk_fraction:.2%}")

    if failed_checks:
        raise SystemExit("Chat generation sanity checks failed: " + "; ".join(failed_checks))
    print("Chat generation sanity checks passed; semantic quality still requires human review.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
