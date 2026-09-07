#!/usr/bin/env python3
"""Generate a large deterministic synthetic corpus for CatAI.

This intentionally does not store the generated corpus in Git: tens of millions
of sentences would make the repository unnecessarily huge. Generate locally with:

    python data/generate_corpus.py --sentences 20000000 --output data/large.txt

The generator uses a fixed seed by default, so the same arguments produce the
same corpus. It mixes several sentence families instead of repeating one line.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

SUBJECTS = [
    "the cat", "a curious cat", "the little cat", "the old cat", "the young cat",
    "the kitten", "a quiet kitten", "the clever kitten", "the fox", "a small bird",
    "the dog", "a friendly dog", "the robot", "a tiny robot", "the learner",
    "the student", "a patient student", "the program", "the computer", "the model",
]
VERBS = [
    "walks", "runs", "waits", "looks", "watches", "learns", "explores", "listens",
    "thinks", "reads", "writes", "searches", "studies", "moves", "rests", "plays",
    "opens", "closes", "follows", "notices", "remembers", "observes", "tests", "builds",
]
OBJECTS = [
    "the room", "the garden", "the window", "the old book", "the bright screen",
    "a new pattern", "a strange sound", "the quiet street", "the wooden door",
    "the small box", "the morning sky", "the warm floor", "a useful example",
    "the next problem", "a simple idea", "the changing world", "the nearby tree",
]
ADJECTIVES = [
    "curious", "quiet", "bright", "small", "gentle", "patient", "careful", "clever",
    "simple", "useful", "interesting", "strange", "familiar", "new", "old", "warm",
]
NOUNS = [
    "pattern", "idea", "answer", "question", "lesson", "example", "signal", "message",
    "window", "garden", "story", "machine", "program", "token", "sequence", "problem",
]
PLACES = [
    "in the room", "near the window", "under the table", "beside the garden",
    "on the quiet street", "inside the small house", "near the old tree", "at the desk",
]


def sentence(rng: random.Random, i: int) -> str:
    kind = i % 10
    subject = rng.choice(SUBJECTS)
    verb = rng.choice(VERBS)

    if kind == 0:
        return f"{subject.capitalize()} {verb} {rng.choice(OBJECTS)}."
    if kind == 1:
        return f"{subject.capitalize()} {verb} {rng.choice(OBJECTS)} {rng.choice(PLACES)}."
    if kind == 2:
        return f"The {rng.choice(ADJECTIVES)} {rng.choice(NOUNS)} becomes easier to understand with practice."
    if kind == 3:
        return f"Learning a {rng.choice(NOUNS)} takes time, attention, and repeated practice."
    if kind == 4:
        return f"When {subject} {verb}, it can discover {rng.choice(OBJECTS)}."
    if kind == 5:
        return f"A useful {rng.choice(NOUNS)} can connect one {rng.choice(NOUNS)} to another."
    if kind == 6:
        return f"The {rng.choice(ADJECTIVES)} {rng.choice(NOUNS)} is followed by a {rng.choice(ADJECTIVES)} {rng.choice(NOUNS)}."
    if kind == 7:
        return f"{subject.capitalize()} sees {rng.choice(OBJECTS)}, pauses, and then {rng.choice(VERBS)} again."
    if kind == 8:
        return f"Small steps can turn a difficult {rng.choice(NOUNS)} into a manageable one."
    return f"A good learner asks a question, studies the evidence, and tries another approach."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentences", type=int, default=20_000_000)
    parser.add_argument("--output", type=Path, default=Path("data/large.txt"))
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    if args.sentences < 1:
        raise SystemExit("--sentences must be positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    with args.output.open("w", encoding="utf-8", buffering=1024 * 1024) as f:
        for i in range(args.sentences):
            f.write(sentence(rng, i))
            f.write("\n")

    print(f"Wrote {args.sentences:,} sentences to {args.output}")


if __name__ == "__main__":
    main()
