# catai

A small decoder-only language model experiment ("maybe like a cat..."). The recommended CI training workflow uses BPE by default; the standalone CLI still offers the character tokenizer for lightweight experiments.

## Architecture (current)

- **Tokenizer**: deterministic BPE subword tokenizer by default; the printable-ASCII character tokenizer remains available for experiments
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

# Build a curated real-text corpus
python -m pip install -e ".[data]"
python data/download_corpus.py --output data/corpus.txt --target-chars 100000000 --fineweb-edu-chars 15000000

# Train a tiny BPE model with a 20M-token budget
python -m catai --corpus data/corpus.txt --checkpoint catai.pt --model-size tiny --tokenizer bpe --vocab-size 512 --max-tokens 200000000 --validation-split 0.05

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


## Auxiliary emotion-recognition data (GoEmotions)

GoEmotions is a **classification dataset**, not a conversational response
dataset. Its examples teach the model to output labels such as `Emotions:
curiosity`; those labels are not appropriate targets for ordinary chat. It is
therefore excluded from the default conversational SFT mixture and retained
as a separate `emotions_only` preparation experiment. Every record preserves
`dataset_source`, `dataset_license`, and the original multi-label annotations.

Prepare it locally:

```bash
python -m pip install pyarrow
python data/prepare_emotion_datasets.py --source go_emotions --output data/go_emotions_emotion.jsonl
python data/merge_emotion_datasets.py --go-emotions data/go_emotions_emotion.jsonl
```

To prepare the classifier data without training CatAI's chat model, run
**Actions → CI → Run workflow** and select `training_mode: emotions_only`.

DailyDialog (CC BY-NC-SA 4.0) and EmpatheticDialogues (CC BY-NC 4.0) converters
remain available only for experiments with separate permission; neither is
used in the default workflow because of its non-commercial restrictions.

## Updated base-training data and budget

The Actions workflow defaults to the **tiny** preset (about 0.92M parameters)
and a 200M-token pretraining cap (about 217.7 input tokens per parameter for
the 512-token BPE vocabulary). After the 5% validation split, that is roughly
206.8 training tokens per parameter. The trainer checks the actual model
vocabulary, parameter count, train split, and presence of document EOS markers
before starting; SFT checks the recorded training metadata again. A rough
20-token compute-optimal reference target:

| Preset | Parameters | 20-token target |
|---|---:|---:|
| tiny | 918,656 | 20M tokens |
| small | 6,557,952 | 131M tokens |
| base | 38,810,112 | 776M tokens |
| medium | 113,664,768 | 2.27B tokens |

These are scaling-law reference points, not a quality guarantee or universal
rule for small models. CatAI logs the model's actual parameter count, training
token count, and achieved ratio.

The corpus combines:
- English RedPajama-V2 `head_middle` documents filtered using published
  Gopher-style quality signals.
- RedPajama's duplicate-ID Parquet artifacts plus a local SHA-256 pass for
  additional exact deduplication.
- A bounded 150M-character FineWeb-Edu sample, filtered to score >= 3 and
  language_score >= 0.8. It is placed first so it appears in the capped
  training-token prefix.

Install corpus dependencies using `python -m pip install -e ".[data]"`.
FineWeb-Edu is ODC-BY-1.0 and remains subject to Common Crawl's terms. See the
[FineWeb-Edu dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
and [RedPajama-V2 paper](https://papers.nips.cc/paper/2024/file/d34497330b1fd6530f7afd86d0df9f76-Paper-Datasets_and_Benchmarks_Track.pdf).

## Additional SFT sources

The default SFT sampling targets emphasize **general assistant
conversations** and keep specialist tasks out of the chat model:

| Source | Target share | Declared dataset license |
|---|---:|---|
| OpenAssistant 1 + 2 | 57% | Apache-2.0 |
| UltraChat 200k | 25% | MIT |
| Smol-SmolTalk | 10% | Apache-2.0 |
| Blended Skill Talk | 5% | CC BY 4.0 |
| CatAI personality examples | 3% | CatAI-authored |

CatAI samples up to 5,000 UltraChat 200k `train_sft` conversations and 2,500
Smol-SmolTalk examples. The sources are compacted and checked against the
model context; their loaded counts are not the same as their trainable counts.
Generated JSONL retains source and declared-license metadata.

GoEmotions is reserved for the separate `emotions_only` experiment because
its target is an emotion label, not a natural conversational response.
GSM8K remains available through the optional MIT-licensed adapter for a
separately evaluated math-specialist experiment, but its worked solutions are
not included in general conversational SFT. This avoids teaching a small
general-purpose model to answer ordinary questions with classification labels
or unrelated step-by-step math traces.

SFT now constructs a held-out validation sample using the same source ratios as
the training sampler and reports loss per supervised assistant token. This
prevents a large raw dataset from dominating validation merely because it has
more rows, when its actual training share is small.

The pretraining corpus writer adds a dedicated EOS token between documents so
the model learns where unrelated source documents end, instead of being trained
to continue seamlessly from the end of one web page into the next.

See [Dataset licenses and provenance](docs/DATASET_LICENSES.md) before
redistributing datasets or model artifacts. A license label in a dataset card
is evidence of the dataset publisher's stated terms, not a guarantee that every
underlying third-party text has been independently rights-cleared. In
particular, the pretraining corpus includes Common Crawl-derived material:
RedPajama-V2 directs users to Common Crawl's terms, and FineWeb-Edu is
ODC-BY-1.0 while remaining subject to those source terms. Do not treat the
entire CatAI corpus or resulting checkpoint as covered by MIT merely because
some SFT datasets are MIT-licensed.

The default training mix and sampling budget are starting points, not a
quality guarantee. The previous full run showed that low training loss and a
green workflow do not prove coherent generation: its base-model sample was
already incoherent before SFT. Compare the fixed prompts and task-specific
scores after every fresh full-pretraining run; SFT cannot reliably teach
foundational language patterns that the base checkpoint has not learned.
