# Dataset licenses and provenance

This file records the licenses stated by the dataset publishers for the exact
sources used by CatAI's default training workflow. It is a provenance aid, not
legal advice or a guarantee that every underlying text is free of third-party
rights.

## Default supervised fine-tuning sources

| Source | Dataset license stated by publisher | Default use |
|---|---|---|
| OpenAssistant OASST1 | Apache-2.0 | Yes |
| OpenAssistant OASST2 | Apache-2.0 | Yes |
| GoEmotions | Apache-2.0 | Yes |
| Smol-SmolTalk | Apache-2.0 | Yes |
| Blended Skill Talk | CC BY 4.0 | Yes, with attribution |
| UltraChat 200k | MIT | Yes |
| GSM8K | MIT | Yes |
| CatAI personality examples | Authored for CatAI | Yes |

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

## Pretraining sources and important qualifications

| Source | Published terms | Qualification |
|---|---|---|
| FineWeb-Edu | ODC-BY-1.0 | Also subject to Common Crawl terms and underlying source rights |
| RedPajama-V2 | Common Crawl Foundation Terms of Use for the data; Apache-2.0 applies to processing code | The dataset card does not declare the text itself MIT or Apache-2.0 |

- [FineWeb-Edu dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- [ODC-By 1.0 license](https://opendatacommons.org/licenses/by/)
- [RedPajama-V2 dataset card and license notice](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2)
- [Common Crawl Terms of Use](https://commoncrawl.org/terms-of-use)

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
