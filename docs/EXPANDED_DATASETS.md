# Expanded Hugging Face chat and emotion datasets

CatAI's dataset-expansion job prepares a bounded, reproducible sample from the
13 additional Hugging Face datasets selected for the November 2026 expansion.
Normal `full` and `sft_only` runs prepare this data automatically and train on
the approved chat subset. The separate `datasets_only` mode prepares the full
shortlist without starting model training. The default sample cap is 1,000
examples per source, with seed 42. The manifest records prepared counts,
publisher-stated terms, source repositories, and preparation errors.

## Catgirl and character-persona chat

| Dataset | Hugging Face repository | Publisher-stated terms | Preparation |
|---|---|---|---|
| Cat-v2.8 | [Nix-ai/Cat-v2.8](https://huggingface.co/datasets/Nix-ai/Cat-v2.8) | Apache-2.0 | Included in the shared 15% expanded-chat SFT pool; source system persona prompts are removed |
| catgirl-1m-corpus | [Compactbot/catgirl-1m-corpus](https://huggingface.co/datasets/Compactbot/catgirl-1m-corpus) | Apache-2.0 | Included in the shared 15% expanded-chat SFT pool; `User:` / `Cat:` pairs are parsed into turns |
| Anime Girl Chat Dataset (English) | [coderian/anime-girl-chat-dataset-en](https://huggingface.co/datasets/coderian/anime-girl-chat-dataset-en) | CC BY 4.0 | Included in the shared 15% expanded-chat SFT pool; `user` / `anime_girl` columns are normalized |

## General conversation and instruction datasets

| Dataset | Hugging Face repository | Publisher-stated terms / provenance | Preparation |
|---|---|---|---|
| OpenHermes 2.5 | [teknium/OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5) | No top-level license is declared on the dataset card; it compiles many upstream sources | Research-only sample pending source/license audit |
| UltraChat 200k | [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | MIT | Approved sample; UltraChat is already part of the existing SFT workflow, so do not add this sample a second time |
| WizardLM Evol-Instruct V2 | [WizardLMTeam/WizardLM_evol_instruct_V2_196k](https://huggingface.co/datasets/WizardLMTeam/WizardLM_evol_instruct_V2_196k) | MIT on the HF card; card describes a mixture of Alpaca and ShareGPT-derived data | Research-only sample pending review of upstream/source terms |
| Dolly 15K | [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | CC BY-SA 3.0 | Research-only sample; review attribution and ShareAlike implications for model distribution |
| GPT4All Prompt Generations | [nomic-ai/gpt4all_prompt_generations](https://huggingface.co/datasets/nomic-ai/gpt4all_prompt_generations) | Apache-2.0 on the HF card | Included in the shared 15% expanded-chat SFT pool; retain the original source field |
| EmpatheticDialogues for LLM | [Estwld/empathetic_dialogues_llm](https://huggingface.co/datasets/Estwld/empathetic_dialogues_llm) | The mirror card reports Apache-2.0, but the original [Facebook EmpatheticDialogues](https://huggingface.co/datasets/facebook/empathetic_dialogues) is CC BY-NC 4.0 | Research-only; the upstream non-commercial terms take precedence for this derivative |
| ESCONV for LLM | [Estwld/esconv_llm](https://huggingface.co/datasets/Estwld/esconv_llm) | Apache-2.0 on the HF mirror card; upstream rights/provenance should be rechecked | Research-only pending upstream review |

## Emotion-recognition datasets

These are classification/recognition examples, not natural assistant replies.
They are written to `expanded_emotion.jsonl`, kept separate from conversational
SFT so CatAI does not learn to answer ordinary chat with emotion labels.

| Dataset | Hugging Face repository | Publisher-stated terms / provenance | Preparation |
|---|---|---|---|
| Emotion | [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) | `other` / no clear license grant on the dataset card | Research-only classification sample |
| Many Emotions | [ma2za/many_emotions](https://huggingface.co/datasets/ma2za/many_emotions) | `other`; combines Emotion, DailyDialog, and GoEmotions with different source-level terms | Research-only; preserve each row's upstream dataset and license fields |
| MELD | [declare-lab/MELD](https://huggingface.co/datasets/declare-lab/MELD) | GPL-3.0 in the source repository; includes dialogue from the TV series *Friends* | Research-only classification sample; underlying dialogue rights require separate review |

Many Emotions is not an independent replacement for the existing GoEmotions
pipeline. Its release includes GoEmotions-derived rows and other sources,
including non-commercial DailyDialog data. CatAI samples the publisher's
10,000-row balanced sample configuration, filters it to English, and excludes
rows marked as exact-text label conflicts. The preparation code preserves
per-row provenance and does not move these examples into the default
commercial-capable training mix.

## Output artifacts

- `catai-expanded-chat-datasets`: complete approved chat sample for auditing and separate experiments.
- `catai-expanded-sft-chat-datasets`: approved SFT sample with UltraChat omitted because the main SFT workflow samples UltraChat independently.
- `catai-research-chat-datasets`: chat examples whose licenses or provenance
  need review before use in a distributable model.
- `catai-expanded-emotion-datasets`: emotion-recognition prompts, separate from
  general chat.
- `catai-expanded-dataset-manifest`: source-level counts and preparation errors.

The regular training workflow assigns the SFT-specific expanded-chat sample a
15% target share. The `datasets_only` mode only prepares artifacts and does
not publish a checkpoint. A publisher's license label does not independently
clear every underlying work.
Review `docs/DATASET_LICENSES.md` and the current upstream cards before using
any artifact in a model release.
