# catai

A small character-level language model experiment ("maybe like a cat...").

## Architecture (current)

- **Tokenizer**: fixed printable-ASCII character vocabulary (can spell any name/code from known characters)
- **Position encoding**: Rotary Position Embeddings (RoPE)
- **Normalization**: RMSNorm
- **Feed-forward**: SwiGLU
- **Attention**: causal multi-head self-attention
- **Default context**: 1024 tokens (configurable per checkpoint)
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
python -m catai --corpus data/train.txt --model-size small --sequence-length 1024 --epochs 2

# Generate
python -m catai.generate_cli --checkpoint catai.pt --prompt "The cat"
```

For better quality, prefer real text (e.g. RedPajama / books / code) over pure synthetic templates, and train longer with a validation split.

## OpenAssistant SFT datasets (OASST1 + OASST2)

CatAI's supervised fine-tuning loader uses state-conditioned chat JSONL. Each
record must contain a `state` object plus a `messages` array. The loader
validates the complete state schema, then renders the state together with
CatAI's canonical personality system prompt before SFT.

CatAI's tokenizer already includes a dedicated `<EOS>` token. SFT trains that token as the target immediately after each complete assistant response, and generation stops when it is produced; regression tests cover both the target label and generation stop behavior.

Instruction examples may also include a top-level `tools` array containing
tool names, descriptions, and JSON argument schemas. Those definitions stay
outside `messages`; the SFT loader builds the corresponding system context
from the metadata for each example. That makes tool availability part of the
model's training input without baking tool instructions into a fake `file`
message or duplicating the prompt text across every conversation. At inference
time, the caller supplies the system prompt and only the tools enabled for that
conversation. Tool-call examples should use the same names and argument shapes
the runtime exposes.

The OASST1 and OASST2 converters use the releases' ready-for-export conversation trees, keep English paths by default, map the `prompter` role to CatAI's `user` role, and emit the required `state` + `messages` record shape. Both datasets lack CatAI's emotion/needs/personality annotations, so converted examples receive CatAI's explicit neutral/default state rather than invented annotations. OASST2 uses the same tree schema and quality filters as OASST1.

Generate the two source datasets and a deterministic, deduplicated, balanced mixture with:

```bash
python data/download_oasst1.py --output data/oasst1_chat.jsonl --keep-source
python data/download_oasst2.py --output data/oasst2_chat.jsonl --keep-source
python data/merge_chat_datasets.py --max-examples 4500 --seed 42
```

The default training context is 1024 tokens; the Actions workflow also offers 512 for faster/lower-memory experiments. Checkpoints record their own context length, and the inference runtime uses that saved value, so a 1024-token checkpoint can use the expanded window at inference time.

The merged SFT file defaults to `data/oasst_chat.jsonl`. The example cap is combined across both sources and is split approximately 50/50, with unused quota filled from whichever source has more examples. Selection and output ordering are deterministic for a fixed seed. Exact duplicate conversations are removed, and each record keeps a `dataset_source` field for auditability. The source archives and generated JSONL files are ignored by Git.

For a smaller experiment:

```bash
python data/download_oasst1.py --max-trees 100 --max-examples 1000
```

The generated datasets and downloaded source archives are ignored by Git. The OASST1 ready export contains 10,364 trees and 88,838 messages. The OASST2 ready export contains 13,854 trees and 135,174 messages, including 64,513 English messages. Both official releases are licensed under Apache-2.0. See the [OASST1 dataset card](https://huggingface.co/datasets/OpenAssistant/oasst1) and [OASST2 dataset card](https://huggingface.co/datasets/OpenAssistant/oasst2).
