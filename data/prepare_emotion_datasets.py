#!/usr/bin/env python3
"""Download and normalize one public emotion dataset for CatAI SFT."""
from __future__ import annotations
import argparse
import csv
import io
import tarfile
import urllib.request
import zipfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from catai.emotion_data import (
    daily_dialog_records, empathetic_dialogues_records, go_emotions_records, write_jsonl,
)

GO_EMOTIONS_URL = (
    "https://huggingface.co/datasets/google-research-datasets/go_emotions/"
    "resolve/main/simplified/train-00000-of-00001.parquet"
)
DAILY_DIALOG_URLS = (
    "http://yanran.li/files/ijcnlp_dailydialog.zip",
    "https://github.com/diixo/daily-dialogs-dataset/raw/refs/heads/main/"
    "ijcnlp_dailydialog.zip",
)
EMPATHETIC_DIALOGUES_URL = (
    "https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/"
    "empatheticdialogues.tar.gz"
)

def download(urls: tuple[str, ...], destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size:
        return destination
    last_error: Exception | None = None
    for url in urls:
        request = urllib.request.Request(url, headers={"User-Agent": "CatAI emotion dataset preparation/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                data = response.read()
            if not data:
                raise ValueError(f"download from {url} was empty")
            temporary = destination.with_name(destination.name + ".part")
            temporary.write_bytes(data)
            temporary.replace(destination)
            print(f"Downloaded {destination.name} ({len(data):,} bytes)")
            return destination
        except Exception as exc:
            last_error = exc
            print(f"Could not download {url}: {type(exc).__name__}: {exc}")
    raise RuntimeError(f"could not download {destination.name}") from last_error

def prepare_go_emotions(output: Path, source_dir: Path) -> int:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError(
            "GoEmotions preparation requires pyarrow; install it with python -m pip install pyarrow."
        ) from exc
    source = download((GO_EMOTIONS_URL,), source_dir / "go_emotions_train.parquet")
    records = go_emotions_records(parquet.read_table(source).to_pylist())
    return write_jsonl(str(output), records)

def _find_zip_member(archive: zipfile.ZipFile, suffix: str) -> str:
    matches = [name for name in archive.namelist() if name.endswith(suffix)]
    if not matches:
        raise ValueError(f"DailyDialog archive is missing {suffix}")
    return matches[0]

def prepare_daily_dialog(output: Path, source_dir: Path) -> int:
    source = download(DAILY_DIALOG_URLS, source_dir / "daily_dialog.zip")
    with zipfile.ZipFile(source) as outer:
        inner_name = _find_zip_member(outer, "train.zip")
        inner_bytes = outer.read(inner_name)
    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as train_zip:
        dialog_name = _find_zip_member(train_zip, "train/dialogues_train.txt")
        emotion_name = _find_zip_member(train_zip, "train/dialogues_emotion_train.txt")
        dialog_lines = train_zip.read(dialog_name).decode("utf-8").splitlines()
        emotion_lines = train_zip.read(emotion_name).decode("utf-8").splitlines()
    if len(dialog_lines) != len(emotion_lines):
        raise ValueError(
            f"DailyDialog train files disagree: {len(dialog_lines)} dialogues vs {len(emotion_lines)} emotion rows"
        )
    dialogues = []
    for line_number, (dialog_line, emotion_line) in enumerate(zip(dialog_lines, emotion_lines), start=1):
        turns = dialog_line.rstrip().split("__eou__")
        emotions = emotion_line.split()
        if turns and not turns[-1].strip():
            turns.pop()
        turns = [turn.strip() for turn in turns]
        if len(turns) != len(emotions):
            raise ValueError(
                f"DailyDialog train row {line_number} has {len(turns)} turns and {len(emotions)} emotion labels"
            )
        dialogues.append({"dialog": turns, "emotion": [int(label) for label in emotions]})
    return write_jsonl(str(output), daily_dialog_records(dialogues))

def prepare_empathetic_dialogues(output: Path, source_dir: Path) -> int:
    source = download((EMPATHETIC_DIALOGUES_URL,), source_dir / "empathetic_dialogues.tar.gz")
    with tarfile.open(source, mode="r:gz") as archive:
        matches = [
            member for member in archive.getmembers()
            if member.isfile() and member.name.endswith("/train.csv")
        ]
        if not matches:
            raise ValueError("EmpatheticDialogues archive is missing empatheticdialogues/train.csv")
        stream = archive.extractfile(matches[0])
        if stream is None:
            raise ValueError("could not read EmpatheticDialogues train.csv")
        text = stream.read().decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text)))
    return write_jsonl(str(output), empathetic_dialogues_records(rows))

def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare one source for CatAI's merged emotional dataset.")
    parser.add_argument("--source", choices=("go_emotions", "daily_dialog", "empathetic_dialogues"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, default=Path("data/emotion_sources"))
    args = parser.parse_args()
    if args.source == "go_emotions":
        count = prepare_go_emotions(args.output, args.source_dir)
    elif args.source == "daily_dialog":
        count = prepare_daily_dialog(args.output, args.source_dir)
    else:
        count = prepare_empathetic_dialogues(args.output, args.source_dir)
    if count < 1:
        raise SystemExit(f"{args.source} produced no usable examples")
    print(f"Prepared {count:,} {args.source} examples at {args.output}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
