# Dataset licenses and provenance

This file records the licenses stated by the dataset publishers for the exact
sources used by CatAI's default training workflow. It is a provenance aid, not
legal advice or a guarantee that every underlying text is free of third-party
rights.

## Supervised fine-tuning sources and modes

| Source | Dataset license stated by publisher | Default use |
|---|---|---|
| OpenAssistant OASST1 | Apache-2.0 | Yes |
| OpenAssistant OASST2 | Apache-2.0 | Yes |
| GoEmotions | Apache-2.0 | No — separate classification experiment |
| Smol-SmolTalk | Apache-2.0 | Yes |
| Blended Skill Talk | CC BY 4.0 | Yes, with attribution |
| UltraChat 200k | MIT | Yes |
| GSM8K | MIT | No — optional math-specialist experiment |
| CatAI personality examples | Authored for CatAI | Yes |

The default general conversational SFT mixture uses OpenAssistant (57%),
UltraChat 200k (25%), Smol-SmolTalk (10%), Blended Skill Talk (5%), and
CatAI-authored personality examples (3%). Its validation sample follows those
same source ratios, and the reported loss is weighted by supervised
assistant-token count.

GoEmotions (Apache-2.0) is reserved for a separate emotion-classification
experiment and is not mixed into general chat SFT. GSM8K (MIT) remains an
optional math-specialist dataset and is not in the default chat mixture.

GoEmotions contains label-prediction examples such as `Emotions: curiosity`.
Those are useful for an emotion-classification experiment, but are deliberately
not mixed into general chat SFT. GSM8K remains an optional math-specialist
source; its step-by-step worked answers are not part of the default assistant
mixture.

Official dataset pages:

- [OASST1](https://huggingface.co/datasets/OpenAssistant/oasst1)
- [OASST2](https://huggingface.co/datasets/OpenAssistant/oasst2)
- [GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions)
- [Smol-SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk)
- [Blended Skill Talk](https://parl.ai/projects/blended_skill_talk/)
- [UltraChat 200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)
- [GSM8K](https://huggingface.co/datasets/openai/gsm8k)

The UltraChat and GSM8K dataset cards declare MIT. The converters preserve
source identifiers and the declared license in the generated JSONL records.
UltraChat was created from model-generated conversations; a declared dataset
license is not an independent rights audit of every prompt or response.

## Expanded dataset shortlist (optional preparation only)

The `datasets_only` workflow prepares bounded samples from the additional
Hugging Face sources documented in [EXPANDED_DATASETS.md](EXPANDED_DATASETS.md).
This preparation track is separate from the default SFT mixture. The terms
below are the publisher-stated terms or known upstream restrictions, not a
legal clearance for model distribution.

| Source | Stated terms / provenance | Prepared output |
|---|---|---|
| Cat-v2.8 | Apache-2.0 | Expanded chat |
| catgirl-1m-corpus | Apache-2.0 | Expanded chat |
| Anime Girl Chat Dataset (English) | CC BY 4.0 | Expanded chat |
| OpenHermes 2.5 | No top-level license declared; composite of upstream sources | Research chat only |
| UltraChat 200k | MIT; already used by the default SFT workflow | Expanded chat sample, do not double-count |
| WizardLM Evol-Instruct V2 | MIT on HF card; mixed upstream provenance | Research chat only pending review |
| Dolly 15K | CC BY-SA 3.0 | Research chat only pending review |
| GPT4All Prompt Generations | Apache-2.0 on HF card; preserve source provenance | Expanded chat |
| EmpatheticDialogues | Original upstream is CC BY-NC 4.0, despite a permissive label on one reformatted mirror | Research chat only |
| ESCONV for LLM | Apache-2.0 on HF mirror card; upstream terms need review | Research chat only |
| Emotion | HF card reports `other`, without a clear license grant | Emotion-recognition only |
| Many Emotions | `other`; combines sources with different terms, including non-commercial data | Emotion-recognition only; preserve row-level provenance |
| MELD | GPL-3.0 in the source repository; includes *Friends* dialogue | Emotion-recognition research only |

Emotion-label datasets are not merged into general conversational SFT. The
workflow produces separate approved-chat, research-chat, and emotion
artifacts, with source and license metadata on every example. A research-only
label means review is needed before using the data in a model intended for
distribution; it does not mean the source is automatically unusable.

## Pretraining sources and important qualifications

| Source | Published terms | Qualification |
|---|---|---|
| FineWeb-Edu | ODC-BY-1.0 | Also subject to Common Crawl terms and underlying source rights |
| RedPajama-V2 | Common Crawl Foundation Terms of Use for the data; Apache-2.0 applies to processing code | The dataset card does not declare the text itself MIT or Apache-2.0 |

- [FineWeb-Edu dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- [ODC-By 1.0 license](https://opendatacommons.org/licenses/by/)
- [RedPajama-V2 dataset card and license notice](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2)
- [Common Crawl Terms of Use](https://commoncrawl.org/terms-of-use)

The pretraining downloader writes a dedicated EOS boundary between documents;
this is model-training structure, not a new dataset or a license change.

Do not describe the complete pretraining corpus as MIT-licensed or assume that
the license on a dataset's processing code licenses the underlying text.
Attribution and source-specific terms must be checked before redistributing
corpus extracts or model artifacts.

## Excluded from the default SFT workflow

These converters remain available for experiments where separate permissions
have been obtained, but the default workflow does not download, merge, or train
on them:

| Source | Dataset license stated by publisher | Reason excluded |
|---|---|---|
| DailyDialog | CC BY-NC-SA 4.0 | Non-commercial restriction |
| EmpatheticDialogues | CC BY-NC 4.0 | Non-commercial restriction |

Official cards: [DailyDialog](https://huggingface.co/datasets/li2017dailydialog/daily_dialog) and
[EmpatheticDialogues](https://huggingface.co/datasets/facebook/empathetic_dialogues).

## Operational policy

1. Do not add a source to the default workflow based only on public download
   availability. Inspect the dataset card and source terms first.
2. Prefer MIT, Apache-2.0, CC0, and attribution-only licenses such as CC BY 4.0
   where the dataset publisher clearly applies the license to the dataset.
3. Exclude licenses with non-commercial, no-derivatives, or other use
   restrictions from the default commercial-capable mixture unless separate
   permission is documented.
4. Keep the original source, declared license, and source URL in generated
   records or an accompanying manifest.
5. Re-check the live dataset card and upstream terms before each new release.
   A license field is a statement by the dataset publisher, not a warranty that
   underlying third-party works are rights-cleared in every jurisdiction.

CatAI's model code license and dataset licenses are separate. The repository's
software license does not override the licenses or terms of training data.
