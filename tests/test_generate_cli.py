import torch

from catai.checkpoint import save_checkpoint
from catai.generate_cli import build_parser, main
from catai.model import CatAI
from catai.tokenizer import CharTokenizer


def test_parser_accepts_generation_options():
    args = build_parser().parse_args(
        [
            "--checkpoint", "model.pt",
            "--prompt", "Cats",
            "--max-new-tokens", "12",
            "--temperature", "0.8",
            "--device", "cpu",
        ]
    )
    assert args.checkpoint == "model.pt"
    assert args.prompt == "Cats"
    assert args.max_new_tokens == 12
    assert args.temperature == 0.8
    assert args.device == "cpu"


def test_generate_cli_loads_checkpoint_and_prints_text(tmp_path, capsys):
    torch.manual_seed(0)
    tokenizer = CharTokenizer(("a", "b", "c"))
    model = CatAI(vocab_size=3, max_seq_len=8, d_model=8, n_heads=2, n_layers=1)
    optimizer = torch.optim.AdamW(model.parameters())
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(
        checkpoint,
        model,
        optimizer,
        step=1,
        metadata={
            "model": {
                "max_seq_len": 8,
                "d_model": 8,
                "n_heads": 2,
                "n_layers": 1,
            },
            "tokenizer": {"type": "char", "vocabulary": list(tokenizer.vocabulary)},
        },
    )

    assert main(
        [
            "--checkpoint", str(checkpoint),
            "--prompt", "a",
            "--max-new-tokens", "2",
            "--device", "cpu",
        ]
    ) == 0
    output = capsys.readouterr().out.strip()
    assert output.startswith("a")
    assert len(output) == 3
