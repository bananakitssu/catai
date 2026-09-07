from catai.checkpoint import load_training_metadata
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
            "--device", "cpu",
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
    assert args.device == "cpu"


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
    assert args.batch_size == 32
    assert args.learning_rate == 3e-4
    assert args.grad_clip == 1.0
    assert args.sequence_length == 32
    assert args.d_model == 128
    assert args.heads == 4
    assert args.layers == 4
    assert args.device is None


def test_cli_trains_and_writes_checkpoint(tmp_path):
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
