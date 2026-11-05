# catai

A small decoder-only language model experiment ("maybe like a cat...") with BPE as the default tokenizer.

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
python -m catai --corpus data/corpus.txt --checkpoint catai.pt --model-size tiny --tokenizer bpe --vocab-size 512 --max-tokens 20000000 --validation-split 0.05

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


## Emotion-aware SFT dataset (GoEmotions + DailyDialog + EmpatheticDialogues)

The CI pipeline prepares a unified, source-attributed emotional dataset from the
training splits of three public datasets:

- **GoEmotions** supplies fine-grained, multi-label emotion-recognition examples.
  These are trained as explicit classification tasks; CatAI is not given invented
  assistant replies to isolated Reddit comments.
- **DailyDialog** supplies actual adjacent dialogue turns. The emotion annotation
  on a user turn is used as context for the real next-speaker response.
- **EmpatheticDialogues** supplies speaker/listener conversations. The listener's
  real response is paired with the speaker's preceding turn and the dialogue's
  annotated emotional context.

The normalized output is \`data/catai_emotion.jsonl\`. Every record retains
\`dataset_source\`, \`dataset_license\`, and emotion labels for provenance. The
merge is deterministic and removes duplicate conversations. Generated JSONL
files and downloaded archives are ignored by Git.

Run the preparation locally:

\`\`\`bash
python -m pip install pyarrow
python data/prepare_emotion_datasets.py --source go_emotions --output data/go_emotions_emotion.jsonl
python data/prepare_emotion_datasets.py --source daily_dialog --output data/daily_dialog_emotion.jsonl
python data/prepare_emotion_datasets.py --source empathetic_dialogues --output data/empathetic_dialogues_emotion.jsonl
python data/merge_emotion_datasets.py
\`\`\`

SFT samples from the merged emotion corpus alongside OpenAssistant and CatAI's
personality examples. The workflow caps the number of draws per epoch so that a
large source archive does not turn one SFT epoch into an unbounded multi-hour job;
increase \`--samples-per-epoch\` when a larger training budget is available.

**Licensing warning:** GoEmotions is Apache-2.0, but DailyDialog is CC BY-NC-SA
4.0 and EmpatheticDialogues is CC BY-NC 4.0. The combined emotional corpus
therefore contains non-commercial source material. Do not use the merged corpus
or assume resulting checkpoints are cleared for commercial use without obtaining
the required permissions. See the
[GoEmotions dataset card](https://huggingface.co/datasets/google-research-datasets/go_emotions),
[DailyDialog dataset card](https://huggingface.co/datasets/li2017dailydialog/daily_dialog),
and [EmpatheticDialogues dataset card](https://huggingface.co/datasets/facebook/empathetic_dialogues).


To validate and generate just the emotion corpus without launching any model
training, manually run **Actions → CI → Run workflow** and select
\`training_mode: emotions_only\`. The three source jobs run in parallel, followed
by the merge and validation job; base training and both SFT stages are skipped.


## Updated base-training data and budget

The Actions workflow defaults to the **tiny** preset (about 0.92M parameters)
and 20M pretraining tokens, approximately 21.8 tokens per parameter. It rejects
budgets below 10 tokens per parameter. A rough 20-token target:

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
- A bounded 15M-character FineWeb-Edu sample, filtered to score >= 3 and
  language_score >= 0.8. It is placed first so it appears in the capped
  training-token prefix.

Install corpus dependencies using `python -m pip install -e ".[data]"`.
FineWeb-Edu is ODC-BY-1.0 and remains subject to Common Crawl's terms. See the
[FineWeb-Edu dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
and [RedPajama-V2 paper](https://papers.nips.cc/paper/2024/file/d34497330b1fd6530f7afd86d0df9f76-Paper-Datasets_and_Benchmarks_Track.pdf).

## Additional SFT sources

The default SFT sampling targets are:

| Source | Target share |
|---|---:|
| OpenAssistant 1 + 2 | 30% |
| CatAI personality examples | 20% |
| Emotion-aware examples | 30% |
| Smol-SmolTalk | 10% |
| Blended Skill Talk | 10% |

Smol-SmolTalk is adapted for models below 1B parameters. CatAI takes up to
1,000 shorter examples and retains the upstream source label; its dataset card
lists Apache-2.0. Blended Skill Talk adds up to 1,000 training conversations:
the free participant becomes the user and the guided participant becomes the
assistant. It is licensed CC BY 4.0. Both converters preserve dataset/license
provenance in each record.

The existing emotion mixture still includes DailyDialog (CC BY-NC-SA 4.0) and
EmpatheticDialogues (CC BY-NC 4.0). The combined emotional corpus therefore
contains non-commercial source material; do not assume the resulting checkpoint
is commercially cleared without separate permission and review.

The added dialogue sources start with small samples. Compare validation loss
and the same CatAI evaluation prompts against the previous 45%/20%/35% SFT
baseline before raising their shares or sample caps.
