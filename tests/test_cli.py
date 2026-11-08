from catai.checkpoint import load_checkpoint_metadata, load_training_metadata
from catai.cli import build_parser, main


def test_parser_accepts_training_options():
    args = build_parser().parse_args(
        [
            "--corpus", "data/tiny.txt",
            "--checkpoint", "model.pt",
            "--epochs", "3",
            "--batch-size", "8",
            "--learning-rate", "0.001",
            "--grad-clip", "0.5",
            "--sequence-length", "16",
            "--d-model", "32",
            "--heads", "2",
            "--layers", "1",
            "--model-size", "base",
            "--tokenizer", "char",
            "--vocab-size", "128",
            "--max-tokens", "1000",
            "--minimum-tokens-per-parameter", "20",
            "--validation-split", "0.05",
            "--patience", "2",
            "--device", "cpu",
            "--window-stride", "8",
        ]
    )
    assert args.corpus == "data/tiny.txt"
    assert args.checkpoint == "model.pt"
    assert args.epochs == 3
    assert args.batch_size == 8
    assert args.learning_rate == 0.001
    assert args.grad_clip == 0.5
    assert args.sequence_length == 16
    assert args.d_model == 32
    assert args.heads == 2
    assert args.layers == 1
    assert args.model_size == "base"
    assert args.tokenizer == "char"
    assert args.vocab_size == 128
    assert args.max_tokens == 1000
    assert args.minimum_tokens_per_parameter == 20
    assert args.validation_split == 0.05
    assert args.patience == 2
    assert args.device == "cpu"
    assert args.window_stride == 8


def test_parser_requires_corpus():
    parser = build_parser()
    try:
        parser.parse_args([])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected argparse to reject missing corpus")


def test_parser_defaults_are_stable():
    args = build_parser().parse_args(["--corpus", "corpus.txt"])
    assert args.checkpoint == "catai.pt"
    assert args.epochs == 1
    assert args.batch_size == 8
    assert args.learning_rate == 3e-4
    assert args.grad_clip == 1.0
    assert args.sequence_length == 1024
    assert args.d_model == 128
    assert args.heads == 4
    assert args.layers == 4
    assert args.model_size == "tiny"
    assert args.tokenizer == "char"
    assert args.vocab_size == 256
    assert args.max_tokens is None
    assert args.minimum_tokens_per_parameter == 0.0
    assert args.validation_split == 0.0
    assert args.patience is None
    assert args.device is None
    assert args.window_stride is None


def test_cli_trains_and_writes_checkpoint(tmp_path, capsys):
    checkpoint = tmp_path / "catai.pt"
    exit_code = main(
        [
            "--corpus", "data/tiny.txt",
            "--checkpoint", str(checkpoint),
            "--epochs", "1",
            "--batch-size", "8",
            "--sequence-length", "16",
            "--d-model", "16",
            "--heads", "2",
            "--layers", "1",
            "--device", "cpu",
        ]
    )

    assert exit_code == 0
    assert checkpoint.exists()
    metadata = load_training_metadata(checkpoint)
    assert metadata["epoch"] == 1
    assert metadata["step"] > 0
    assert isinstance(metadata["loss"], float)
    training = load_checkpoint_metadata(checkpoint)["training"]
    assert training["elapsed_seconds"] > 0
    assert training["tokens_per_second"] > 0
    assert abs(
        training["tokens_per_second"]
        - training["total_tokens"] / training["elapsed_seconds"]
    ) < 1e-6

    output = capsys.readouterr().out
    assert "epoch 1/1" in output
    assert "batch" in output
    assert "loss=" in output
    assert "saved checkpoint" in output


def test_cli_rejects_an_underfunded_training_budget_before_training(tmp_path):
    import pytest

    from catai.tokenizer import EOS_TOKEN

    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        ("banana cat " * 200) + EOS_TOKEN + "\n" + ("dog cat " * 200),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "should-not-exist.pt"

    with pytest.raises(ValueError, match="training budget is too small"):
        main([
            "--corpus", str(corpus),
            "--checkpoint", str(checkpoint),
            "--tokenizer", "bpe",
            "--vocab-size", "8",
            "--sequence-length", "16",
            "--d-model", "16",
            "--heads", "2",
            "--layers", "1",
            "--minimum-tokens-per-parameter", "100",
            "--device", "cpu",
        ])
    assert not checkpoint.exists()
