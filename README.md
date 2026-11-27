# catai

A decoder-only language model experiment ("maybe like a cat..."). The GitHub Actions workflow defaults to the `base` preset with a 2,048-token BPE vocabulary. `small` and `tiny` remain available for shorter experiments, while `medium` is intended for larger token budgets. The standalone CLI also offers the character tokenizer.

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
python data/download_corpus.py --output data/corpus.txt --target-chars 1000000000 --fineweb-edu-chars 150000000

# Train the recommended small BPE model with a 200M-token budget
python -m catai --corpus data/corpus.txt --checkpoint catai.pt --model-size small --tokenizer bpe --vocab-size 2048 --max-tokens 200000000 --validation-split 0.05 --warmup-fraction 0.03 --minimum-learning-rate-ratio 0.1

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

To run the repository's compile checks and test suite without starting any
training or dataset-preparation jobs, select `training_mode: tests_only`.

DailyDialog (CC BY-NC-SA 4.0) and EmpatheticDialogues (CC BY-NC 4.0) converters
remain available only for experiments with separate permission; neither is
used in the default workflow because of its non-commercial restrictions.

## Expanded Hugging Face chat and emotion datasets

The `datasets_only` Actions mode remains available to inspect and download bounded samples from 13 additional Hugging Face datasets: three catgirl/persona chat datasets, general conversation and instruction datasets, and three emotion-recognition datasets. It writes separate approved-chat, research-chat, and emotion artifacts, plus a source/license manifest.

Normal `full` and `sft_only` training runs now prepare the approved chat subset automatically and include it in SFT at a 15% target share. The SFT-specific artifact excludes UltraChat because the standard workflow already samples UltraChat separately. Research-only and emotion-classification data remain excluded from ordinary assistant-response training, and all records retain source/license provenance.

See [the expanded dataset catalog](docs/EXPANDED_DATASETS.md) for every
Hugging Face repository, license notes, and the generated artifacts. See
[Dataset licenses and provenance](docs/DATASET_LICENSES.md) before using
research-only samples in a model intended for distribution.

To prepare the complete shortlist without training, run **Actions → CI → Run workflow** and select `training_mode: datasets_only`. Full and SFT-only training use the same bounded sample preparation automatically. Samples are not committed to Git.

For GPU training, see [the Kaggle GPU guide](docs/KAGGLE_GPU_TRAINING.md) and the two notebooks under `notebooks/`. The base-pretraining notebook retains the full 850M-token budget. To prepare the SFT inputs without running CPU fine-tuning, select `training_mode: prepare_sft_data`, provide the run ID of a previous base-checkpoint artifact, and download the resulting `catai-kaggle-sft-inputs` artifact. For a GPU worker that does not require Kaggle, see [the Modal GPU training guide](docs/MODAL_GPU_TRAINING.md) and `modal/catai_train.py`.

## Updated base-training data and budget

The Actions workflow defaults to the **base** preset (39,596,544 parameters), a 2,048-token BPE vocabulary, and an 850M-token pretraining cap. After the 5% validation split, a fully populated 850M-token run provides about 20.4 training tokens per parameter, meeting the workflow's 20-token training-data threshold. The hosted corpus request is intentionally larger than the token cap because source characters do not map one-to-one to BPE tokens and filters reject some rows; the CLI checks the actual encoded corpus and refuses to train when the minimum is not met. Base training uses 3% linear warmup followed by cosine decay to 10% of the peak learning rate. The base preset is split across 128 resumable Actions jobs, with each intermediate stage processing half as many batches as in the 64-stage layout while preserving the full training budget. The hosted workflow exposes `tiny`, `small`, and `base`; the larger `medium` preset remains available in the Python CLI for hardware and token budgets beyond the hosted runner design. The trainer checks the actual model
vocabulary, parameter count, train split, and presence of document EOS markers
before starting; SFT checks the recorded training metadata again. A rough
20-token compute-optimal reference target:

| Preset | Parameters | 20-token target |
|---|---:|---:|
| tiny | 1,115,264 | 22M tokens |
| small | 6,951,168 | 139M tokens |
| base | 39,596,544 | 792M tokens |
| medium | 114,844,416 | 2.30B tokens |

These are scaling-law reference points, not a quality guarantee or universal
rule for small models. CatAI logs the model's actual parameter count, training
token count, and achieved ratio.

The hosted Base corpus build requests up to 1.89B source characters across
four sources: 1.4B RedPajama-V2, 210M FineWeb-Edu, 140M Cosmopedia, and
140M Common Corpus characters. Common Corpus is additionally capped at
7,000 accepted documents. These are source-character ceilings, not a
guaranteed token count; source filtering and availability can reduce the
actual output. The generated manifest records the actual characters and
documents accepted from every source. The Common Corpus downloader defaults to
7,000 accepted documents because the previous 1,000-document default stopped
well before its 100M-character request was filled.

To build the same Base-scale corpus from the command line:

```bash
python data/download_corpus.py --output data/corpus.txt \
  --target-chars 1400000000 --fineweb-edu-chars 210000000 \
  --cosmopedia-chars 140000000 --common-corpus-chars 140000000 \
  --common-corpus-max-documents 7000
```

The corpus combines:
- English RedPajama-V2 `head_middle` documents filtered using published
  Gopher-style quality signals, plus duplicate-ID Parquet filtering and a
  local SHA-256 pass for additional exact deduplication.
- FineWeb-Edu, filtered to score >= 3 and language_score >= 0.8.
- Cosmopedia educational and story subsets, sampled with per-subset limits.
- English Common Corpus documents that pass the explicit per-document
  license allowlist.

Install corpus dependencies using `python -m pip install -e ".[data]"`.
FineWeb-Edu is ODC-BY-1.0 and remains subject to Common Crawl's terms. See the
[FineWeb-Edu dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
and [RedPajama-V2 paper](https://papers.nips.cc/paper/2024/file/d34497330b1fd6530f7afd86d0df9f76-Paper-Datasets_and_Benchmarks_Track.pdf).

## Additional SFT sources

The default SFT sampling targets emphasize **general assistant
conversations** and keep specialist tasks out of the chat model:

| Source | Target share | Declared dataset license |
|---|---:|---|
| OpenAssistant 1 + 2 | 37% | Apache-2.0 |
| UltraChat 200k | 25% | MIT |
| Smol-SmolTalk | 10% | Apache-2.0 |
| Blended Skill Talk | 5% | CC BY 4.0 |
| Approved expanded chat | 15% | Mixed: Apache-2.0 / CC BY 4.0 publisher-stated terms |
| CatAI personality examples | 3% | CatAI-authored |
| CatAI Markdown formatting | 5% | CatAI-authored, Apache-2.0 |

CatAI samples up to 5,000 UltraChat 200k `train_sft` conversations, 2,500
Smol-SmolTalk examples, and up to 1,000 examples each from the approved
expanded chat sources (Cat-v2.8, catgirl-1m-corpus, Anime Girl Chat, and
GPT4All Prompt Generations). The expanded-chat portion is kept separate from
UltraChat to prevent double weighting. A separate 72-example, CatAI-authored
Markdown set teaches headings, nested and numbered lists, tables, checklists,
links, blockquotes, inline code, and fenced code blocks. It is generated by
`data/prepare_markdown_sft.py`, requires no third-party download, and gets a
5% SFT sampling share. Sources are compacted and checked against the model
context; loaded counts are not the same as trainable counts. Generated JSONL
retains source and declared-license metadata.

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

## Contact

**Contact Email:** catai.dev@proton.me
