# Running CatAI training on Modal GPUs

Modal is an optional GPU worker. GitHub Actions remains the CI/orchestration path; this runner executes the repository's CatAI training CLI on a Modal GPU and stores checkpoints in a persistent Modal Volume.

## Security and billing first

- Modal is a third-party cloud provider. Upload only the CatAI corpus, prepared training JSONL files, and model checkpoints needed for training. Do not upload unrelated personal files, GitHub tokens, B2 credentials, or `.env` files.
- Modal documents TLS-protected connections, encryption at rest and in transit, gVisor-based container isolation, and SOC 2 Type II controls. These controls reduce risk; no provider can guarantee that a breach will never happen. Review [Modal's security and privacy documentation](https://modal.com/docs/guide/security) before uploading private data.
- Modal Volumes persist until deleted. Keep a separate backup of important checkpoints and delete the Modal Volume when you no longer need the remote copy.
- Modal Starter currently advertises $30/month in compute credits, then compute is pay-as-you-go. GPU, CPU, memory, storage and network charges can all apply. See [live Modal pricing](https://modal.com/pricing).
- The runner refuses to submit a job unless you pass `--confirm-paid-compute`. That flag is an explicit acknowledgement, not a spending cap. It does not configure a hard budget. The Modal Starter plan does not list environment-level budgets; do not assume credits prevent all charges.
- No GPU job is automatically launched by GitHub Actions. You must authenticate Modal and run the command yourself.

## 1. Install and authenticate

Install the optional dependency from the CatAI repository root:

```bash
python -m pip install modal
modal setup
```

Complete Modal's browser-based authentication. This creates local Modal credentials; do not commit them or paste them into chat. Check Modal's current signup flow before proceeding if you do not want to provide a phone number or payment method.

## 2. Create the persistent training Volume

```bash
modal volume create catai-training
```

The runner mounts this Volume at `/mnt/catai`. Put inputs under its `inputs/` directory. Keep the Volume private to your Modal workspace.

## 3. Upload inputs

For **base pretraining**, upload the exact corpus file produced by CatAI's corpus preparation workflow:

```bash
modal volume put catai-training ./corpus.txt inputs/corpus.txt
```

For **SFT**, upload the base checkpoint and all prepared SFT datasets from the `catai-kaggle-sft-inputs` Actions artifact. The runner expects these exact paths:

```bash
modal volume put catai-training ./catai.pt inputs/catai.pt
modal volume put catai-training ./oasst_chat.jsonl inputs/oasst_chat.jsonl
modal volume put catai-training ./instruction_chat.jsonl inputs/instruction_chat.jsonl
modal volume put catai-training ./smol_smoltalk_chat.jsonl inputs/smol_smoltalk_chat.jsonl
modal volume put catai-training ./blended_skill_talk_chat.jsonl inputs/blended_skill_talk_chat.jsonl
modal volume put catai-training ./ultrachat_200k_chat.jsonl inputs/ultrachat_200k_chat.jsonl
modal volume put catai-training ./expanded_sft_chat.jsonl inputs/expanded_sft_chat.jsonl
modal volume put catai-training ./markdown_sft_chat.jsonl inputs/markdown_sft_chat.jsonl
```

Do not upload the artifact ZIP itself; extract it first. The base checkpoint and prepared datasets remain private.

## 4. Run SFT

Start with the L4 GPU. Modal's published L4 list rate is $0.000222/second (about $0.80/GPU-hour before CPU, memory, storage, network, taxes or credits). T4 is listed at $0.000164/second (about $0.59/GPU-hour). Actual total cost depends on runtime and resource usage.

```bash
modal run modal/catai_train.py --mode sft --gpu L4 --confirm-paid-compute
```

The SFT runner retains the existing workload: 3 epochs, 6,000 samples per epoch, batch size 2, 1,024-token sequence length, learning rate 2e-5, gradient clipping 1.0, 5% validation split, 500 maximum validation examples, all seven input files, all configured dataset sampling ratios, and recovery checkpoints every 50 batches.

## 5. Run base pretraining

Upload `corpus.txt` and run:

```bash
modal run modal/catai_train.py --mode base --gpu L4 --confirm-paid-compute
```

The base runner retains the 850,000,000-token budget, base preset, BPE vocabulary 2,048, sequence length 1,024, batch size 4, one epoch, 5% validation split, 20-token-per-parameter minimum, 3% warmup and 10% learning-rate floor. It runs resumable 15%-of-work segments and commits its recovery checkpoint to the Volume after each segment. Re-run the same command until it reports completion.

## 6. Retrieve outputs

```bash
modal volume get catai-training outputs ./catai-modal-outputs
```

Completed runs export both the PyTorch checkpoint and the NumPy inference bundle. SFT outputs are `catai-sft.pt` and `catai-sft.npz`; base outputs are `catai.pt` and `catai-base.npz`. Recovery checkpoints are also kept in `outputs/`.

## GPU choices

The runner accepts `T4`, `L4`, `A10`, and `A100`. The default is L4. A larger GPU may not make this relatively small model proportionally faster; benchmark actual tokens/second and total runtime before choosing a more expensive option.

## Current limitations

- This is a manual Modal runner, not yet an Actions-triggered Modal backend. GitHub Actions does not receive Modal credentials, and no provider job starts automatically.
- The base job is segmented and resumable. SFT currently runs as one Modal invocation; if it exceeds the six-hour function timeout, it must be resumed from the last committed recovery checkpoint.
- The Modal Volume is committed after the training process exits, including non-zero exits. A hard platform termination before the process reaches that point may lose uncommitted changes; the original source artifact and independent backup remain important.
- Publishing to CatAI's B2 model manifest is intentionally not automated here.
