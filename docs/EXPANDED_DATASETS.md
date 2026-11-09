# Expanded Hugging Face chat and emotion datasets

CatAI's optional \`datasets_only\` workflow prepares a bounded, reproducible sample
from the 13 additional Hugging Face datasets selected for the November 2026
dataset expansion. It writes JSONL artifacts and a manifest; it does not start
model training or commit downloaded data to Git.

Run **Actions → CI → Run workflow** and select \`training_mode: datasets_only\`.
The default sample cap is 1,000 examples per source, with seed 42. The manifest
records prepared counts, publisher-stated terms, source repositories, and any
source that could not be downloaded or converted.

## Catgirl and character-persona chat

| Dataset | Hugging Face repository | Publisher-stated terms | Preparation |
|---|---|---|---|
| Cat-v2.8 | [Nix-ai/Cat-v2.8](https://huggingface.co/datasets/Nix-ai/Cat-v2.8) | Apache-2.0 | Approved chat sample; source system persona prompts are removed |
| catgirl-1m-corpus | [Compactbot/catgirl-1m-corpus](https://huggingface.co/datasets/Compactbot/catgirl-1m-corpus) | Apache-2.0 | Approved chat sample; \`User:\` / \`Cat:\` pairs are parsed into turns |
| Anime Girl Chat Dataset (English) | [coderian/anime-girl-chat-dataset-en](https://huggingface.co/datasets/coderian/anime-girl-chat-dataset-en) | CC BY 4.0 | Approved chat sample; \`user\` / \`anime_girl\` columns are normalized |

## General conversation and instruction datasets

| Dataset | Hugging Face repository | Publisher-stated terms / provenance | Preparation |
|---|---|---|---|
| OpenHermes 2.5 | [teknium/OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5) | No top-level license is declared on the dataset card; it compiles many upstream sources | Research-only sample pending source/license audit |
| UltraChat 200k | [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | MIT | Approved sample; UltraChat is already part of the existing SFT workflow, so do not add this sample a second time |
| WizardLM Evol-Instruct V2 | [WizardLMTeam/WizardLM_evol_instruct_V2_196k](https://huggingface.co/datasets/WizardLMTeam/WizardLM_evol_instruct_V2_196k) | MIT on the HF card; card describes a mixture of Alpaca and ShareGPT-derived data | Research-only sample pending review of upstream/source terms |
| Dolly 15K | [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | CC BY-SA 3.0 | Research-only sample; review attribution and ShareAlike implications for model distribution |
| GPT4All Prompt Generations | [nomic-ai/gpt4all_prompt_generations](https://huggingface.co/datasets/nomic-ai/gpt4all_prompt_generations) | Apache-2.0 on the HF card | Approved bounded sample; retain the original source field |
| EmpatheticDialogues for LLM | [Estwld/empathetic_dialogues_llm](https://huggingface.co/datasets/Estwld/empathetic_dialogues_llm) | The mirror card reports Apache-2.0, but the original [Facebook EmpatheticDialogues](https://huggingface.co/datasets/facebook/empathetic_dialogues) is CC BY-NC 4.0 | Research-only; the upstream non-commercial terms take precedence for this derivative |
| ESCONV for LLM | [Estwld/esconv_llm](https://huggingface.co/datasets/Estwld/esconv_llm) | Apache-2.0 on the HF mirror card; upstream rights/provenance should be rechecked | Research-only pending upstream review |

## Emotion-recognition datasets

These are classification/recognition examples, not natural assistant replies.
They are written to \`expanded_emotion.jsonl\`, kept separate from conversational
SFT so CatAI does not learn to answer ordinary chat with emotion labels.

| Dataset | Hugging Face repository | Publisher-stated terms / provenance | Preparation |
|---|---|---|---|
| Emotion | [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) | \`other\` / no clear license grant on the dataset card | Research-only classification sample |
| Many Emotions | [ma2za/many_emotions](https://huggingface.co/datasets/ma2za/many_emotions) | \`other\`; combines Emotion, DailyDialog, and GoEmotions with different source-level terms | Research-only; preserve each row's upstream dataset and license fields |
| MELD | [declare-lab/MELD](https://huggingface.co/datasets/declare-lab/MELD) | GPL-3.0 in the source repository; includes dialogue from the TV series *Friends* | Research-only classification sample; underlying dialogue rights require separate review |

Many Emotions is not an independent replacement for the existing GoEmotions
pipeline. Its release includes GoEmotions-derived rows and other sources,
including non-commercial DailyDialog data. CatAI samples the publisher's
10,000-row balanced sample configuration, filters it to English, and excludes
rows marked as exact-text label conflicts. The preparation code preserves
per-row provenance and does not move these examples into the default
commercial-capable training mix.

## Output artifacts

- \`catai-expanded-chat-datasets\`: chat examples from the sources marked approved.
- \`catai-research-chat-datasets\`: chat examples whose licenses or provenance
  need review before use in a distributable model.
- \`catai-expanded-emotion-datasets\`: emotion-recognition prompts, separate from
  general chat.
- \`catai-expanded-dataset-manifest\`: source-level counts and preparation errors.

The workflow only prepares and validates samples. It does not alter the
existing default SFT source ratios, publish a checkpoint, or imply that a
publisher's license label independently clears every underlying work.
Review \`docs/DATASET_LICENSES.md\` and the current upstream cards before using
any artifact in a model release.
