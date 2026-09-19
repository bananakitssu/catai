# catai

A small character-level language model experiment ("maybe like a cat...").

## Recent improvements

- **Fixed character vocabulary** – the character tokenizer always includes printable ASCII, so the model can spell arbitrary names/codes (e.g. `BTDPE`, `XQZ_931`) even when those exact strings never appeared in training.
- **Larger default context** – sequence length default raised from 32 → 256.
- **Better generation defaults** – temperature 0.8, top-k 40, milder repetition penalty.

## Quick start

```bash
# Generate a synthetic corpus (or use any UTF-8 text file)
python data/generate_corpus.py --sentences 100000 --output data/train.txt

# Train (tiny model, character tokenizer)
python -m catai --corpus data/train.txt --checkpoint catai.pt --epochs 1

# Generate
python -m catai.generate_cli --checkpoint catai.pt --prompt "The cat"
```

Use `--model-size small` or `--model-size base` and a larger `--sequence-length` (512–1024) when you have more data and compute.
