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

CatAI's supervised fine-tuning loader already accepts chat JSONL with a `messages`
array. The OASST1 converter downloads the ready-for-export conversation trees,
keeps English paths by default, maps OASST1's `prompter` role to CatAI's
`user` role, and adds CatAI's personality system message to every example.

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
