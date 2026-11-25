# Running CatAI SFT on a Kaggle GPU

This is an optional GPU worker for the existing SFT workload. It does **not** lower the training budget or change the model, sequence length, sample count, sampling ratios, batch size, learning rate, clipping, validation split, or epochs. The only execution changes are CUDA, fused AdamW where supported, fused scaled-dot-product attention, and automatic BF16 on GPUs that support it.

## 1. Prepare the training inputs

A completed base-training run must already have produced the `catai-base-checkpoint` artifact.

1. Open **Actions → CI → Run workflow**.
2. Select `training_mode: prepare_sft_data`.
3. Set `base_checkpoint_run_id` to the Actions run ID that contains the `catai-base-checkpoint` artifact.
4. Run the workflow. It prepares the same SFT datasets as the normal pipeline and uploads one artifact named `catai-kaggle-sft-inputs`. It skips the CPU SFT stages.
5. Download that artifact, extract it, and create a **private Kaggle Dataset** named `catai-training-data`. Keep the artifact's `data/` directory intact. The notebook accepts files either at the dataset root or inside `data/`.

The dataset must contain:

- `catai.pt`
- `data/oasst_chat.jsonl`
- `data/instruction_chat.jsonl`
- `data/smol_smoltalk_chat.jsonl`
- `data/blended_skill_talk_chat.jsonl`
- `data/ultrachat_200k_chat.jsonl`
- `data/expanded_sft_chat.jsonl`
- `data/markdown_sft_chat.jsonl`

Do not make this dataset public: it contains the base checkpoint and prepared training data.

## 2. Create the Kaggle notebook

1. Upload or import `notebooks/catai-sft-gpu-kaggle.ipynb` as a Kaggle Notebook.
2. Attach the private `catai-training-data` dataset under the notebook's **Input** panel.
3. In Kaggle **Settings → Secrets**, add `GITHUB_TOKEN` with read-only access to the private `bananakitssu/catai` repository. The notebook uses it only to clone the source; it never prints the token.
4. In the notebook editor's right-side **Settings** pane, open **Session options**:
   - Set **Accelerator** to a GPU.
   - Enable **Internet** if it is off.
5. Run all cells.

The notebook uses the same workflow SFT values:

| Setting | Value |
|---|---:|
| Epochs | 3 |
| Samples per epoch | 6,000 |
| Batch size | 2 |
| Sequence length | 1,024 |
| Learning rate | 0.00002 |
| Gradient clipping | 1.0 |
| Validation split | 5% |
| Maximum validation examples | 500 |
| Minimum pretraining tokens per parameter | 20 |
| Seed | 42 |
| Checkpoint interval | 50 batches |

All existing dataset ratios are retained: personality 0.03, Smol-SmolTalk 0.10, Blended Skill Talk 0.05, UltraChat 0.25, expanded chat 0.15, and Markdown 0.05.

## 3. Outputs and recovery

Kaggle saves these files under `/kaggle/working`:

- `catai-sft.pt` — trained checkpoint
- `catai-sft-recovery.pt` — resumable recovery checkpoint
- `catai-sft.npz` — inference-only NumPy bundle

Download the notebook output files before the Kaggle session expires. To resume later, add the previous `catai-sft-recovery.pt` to the private Kaggle dataset and run the notebook again. The notebook detects it and passes `--resume-from`.

This notebook currently trains and exports the SFT model; it does not publish to B2 automatically. Review and publish the output through the existing deployment process.

## Notes

- Kaggle GPU availability and session limits are controlled by Kaggle. A GPU run is expected to be faster than the CPU path, but a two-hour finish is a target to measure, not a guarantee.
- The base pretraining stage is intentionally not included in this SFT notebook. It requires the full curated pretraining corpus and its own 850-million-token budget; do not substitute a smaller corpus or token budget just to fit a session.
- If the Internet or Accelerator controls are missing, expand the notebook's right-side Settings pane and check **Session options**. Kaggle's notebook editor documents Internet and accelerator as session options. Some account/competition restrictions can hide or disable them.
