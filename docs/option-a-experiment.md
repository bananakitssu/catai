# CatAI Option A: from-scratch experiment

This workflow tests a subword-tokenized CatAI checkpoint without changing the live model.

## First experiment

Run the **CI** workflow manually with these defaults:

- Training mode: `full`
- Tokenizer: `bpe`
- Vocabulary target: 512 plus any additional character coverage needed by the corpus
- Tokenizer merge sample: the first 250,000 characters of the downloaded corpus
- Context length: 512 tokens for pretraining and SFT

The character-level comparison is available only with a 1,024-token context because its character-tokenized system/state prefix does not fit in 512 positions. The first BPE experiment supports 512 or 1,024 positions; 512 is the cheaper default.
- Base training budget: 5,000,000 tokens, one epoch
- Personality share during SFT: 25% of sampled training examples
- Publish model: disabled

The 5M-token budget is a starting point to measure runtime and generation quality, not the final target. The current RedPajama cache is 100 million characters, so a 100M-token run cannot be obtained from this corpus just by raising `--max-tokens`; it needs more source text as well. Larger token budgets also cost more CPU time, so scale them after the first run gives a real throughput measurement.

The current BPE implementation learns its merges from a sample, then encodes training text in chunks. It preserves every character found in the source corpus, even when that makes the actual vocabulary larger than the 512 target; it reserves merge capacity based on a common-character baseline so rare Unicode characters do not crowd out all learned subwords.

## Data mix and filtering

OASST1 remains separate from `data/instruction_chat.jsonl`. The converter keeps English messages, skips messages with a negative review result or deletion flag, filters assistant replies ranked worse than 1 when a rank exists, rejects high-scoring spam/language-mismatch/PII/inappropriate/hate/sexual-content/toxicity labels, and caps paths at four messages. Unranked messages are retained because the dataset does not rank every reply. Rank 0 is best; `rank >= 0` is not a useful quality filter.

SFT splits the two source datasets into training and validation separately. A weighted sampler then draws an expected 25% of each training epoch from the curated personality set, without physically duplicating examples into validation. The exact share in one epoch is stochastic.

## How to evaluate

Keep a fixed prompt set and compare responses from the existing checkpoint and the experimental checkpoint for relevance, readable English, following the question, repetition, and cat personality. Check base-model validation loss within a tokenizer setup; character-level and BPE losses are not directly comparable because their token units differ.

After measuring the first run, increase the budget in steps and record validation loss, tokens per second, runtime, and sample responses. Do not publish the new checkpoint until its outputs are better. The workflow's **Publish model** switch defaults off, so a test run saves artifacts without changing the active B2 model manifest.
