# catai

A small character-level language model experiment ("maybe like a cat...").

## Architecture (current)

- **Tokenizer**: fixed printable-ASCII character vocabulary (can spell any name/code from known characters)
- **Position encoding**: Rotary Position Embeddings (RoPE)
- **Normalization**: RMSNorm
- **Feed-forward**: SwiGLU
- **Attention**: causal multi-head self-attention
- **Default context**: 256 tokens
- **Generation defaults**: temperature 0.8, top-k 40

### Model presets

| Name    | d_model | heads | layers |
|---------|---------|-------|--------|
| tiny    | 128     | 4     | 4      |
| small   | 256     | 8     | 8      |
| base    | 512     | 8     | 12     |
| medium  | 768     | 12    | 16     |

## Quick start

```bash
# Generate a synthetic corpus (or use any UTF-8 text file)
python data/generate_corpus.py --sentences 100000 --output data/train.txt

# Train (tiny model, character tokenizer)
python -m catai --corpus data/train.txt --checkpoint catai.pt --epochs 1

# Or a larger preset
python -m catai --corpus data/train.txt --model-size small --sequence-length 512 --epochs 2

# Generate
python -m catai.generate_cli --checkpoint catai.pt --prompt "The cat"
```

For better quality, prefer real text (e.g. RedPajama / books / code) over pure synthetic templates, and train longer with a validation split.

## OASST1 chat dataset

CatAI's supervised fine-tuning loader uses state-conditioned chat JSONL. Each
record must contain a `state` object plus a `messages` array. The loader
validates the complete state schema, then renders the state together with
CatAI's canonical personality system prompt before SFT.

Instruction examples may also include a top-level `tools` array containing
tool names, descriptions, and JSON argument schemas. Those definitions stay
outside `messages`; the SFT loader builds the corresponding system context
from the metadata for each example. That makes tool availability part of the
model's training input without baking tool instructions into a fake `file`
message or duplicating the prompt text across every conversation. At inference
time, the caller supplies the system prompt and only the tools enabled for that
conversation. Tool-call examples should use the same names and argument shapes
the runtime exposes.

The OASST1 converter downloads the ready-for-export conversation trees, keeps
English paths by default, maps OASST1's `prompter` role to CatAI's `user`
role, and emits the same required `state` + `messages` record shape. OASST1
does not provide CatAI emotion/needs/personality annotations, so its converted
examples receive CatAI's explicit neutral/default state rather than invented
OASST emotion labels.

Generate the dataset with:

```bash
python data/download_oasst1.py --output data/oasst1_chat.jsonl
```

For a smaller experiment:

```bash
python data/download_oasst1.py --max-trees 100 --max-examples 1000
```

The generated dataset and downloaded source archive are ignored by Git. The
OASST1 ready export contains 10,364 trees and 88,838 messages and is licensed
under Apache-2.0.
