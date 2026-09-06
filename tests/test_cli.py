from catai.cli import build_parser


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
