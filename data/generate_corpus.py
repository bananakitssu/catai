#!/usr/bin/env python3
"""Generate a large deterministic synthetic corpus for CatAI.

This intentionally does not store the generated corpus in Git: large corpora
would make the repository unnecessarily huge. Generate locally with:

    python data/generate_corpus.py --sentences 20000000 --output data/large.txt

The generator uses a fixed seed by default, but now mixes a much broader human
vocabulary and many sentence structures so scaling the corpus also increases
linguistic diversity rather than mostly repeating a tiny template set.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

SUBJECTS = [
    "the cat", "a curious cat", "the little cat", "the old cat", "the young cat",
    "the kitten", "a quiet kitten", "the clever kitten", "the fox", "a small bird",
    "the dog", "a friendly dog", "the robot", "a tiny robot", "the learner",
    "the student", "a patient student", "the teacher", "the scientist", "the engineer",
    "the writer", "the reader", "the child", "the family", "the farmer", "the traveler",
    "the musician", "the artist", "the developer", "the researcher", "the computer",
    "the model", "the machine", "a helpful assistant", "an observant person",
]

VERBS = [
    "walks", "runs", "waits", "looks", "watches", "learns", "explores", "listens",
    "thinks", "reads", "writes", "searches", "studies", "moves", "rests", "plays",
    "opens", "closes", "follows", "notices", "remembers", "observes", "tests", "builds",
    "finds", "carries", "holds", "asks", "answers", "explains", "creates", "changes",
    "checks", "compares", "connects", "discovers", "examines", "improves", "uses",
    "needs", "wants", "knows", "sees", "hears", "feels", "tries", "starts", "finishes",
]

OBJECTS = [
    "the room", "the garden", "the window", "the old book", "the bright screen",
    "a new pattern", "a strange sound", "the quiet street", "the wooden door",
    "the small box", "the morning sky", "the warm floor", "a useful example",
    "the next problem", "a simple idea", "the changing world", "the nearby tree",
    "the blue notebook", "a difficult question", "the final answer", "a short message",
    "the computer screen", "a long sentence", "the open file", "a new lesson",
    "the previous result", "a useful tool", "the empty page", "the latest version",
    "a familiar place", "the correct number", "the hidden detail", "a possible solution",
]

ADJECTIVES = [
    "curious", "quiet", "bright", "small", "gentle", "patient", "careful", "clever",
    "simple", "useful", "interesting", "strange", "familiar", "new", "old", "warm",
    "large", "quick", "slow", "strong", "kind", "clear", "common", "different",
    "important", "possible", "correct", "wrong", "early", "late", "local", "recent",
]

NOUNS = [
    "pattern", "idea", "answer", "question", "lesson", "example", "signal", "message",
    "window", "garden", "story", "machine", "program", "token", "sequence", "problem",
    "result", "sentence", "word", "book", "screen", "file", "number", "system",
    "computer", "model", "language", "person", "animal", "place", "reason", "method",
    "task", "project", "memory", "conversation", "question", "solution", "mistake",
]

PLACES = [
    "in the room", "near the window", "under the table", "beside the garden",
    "on the quiet street", "inside the small house", "near the old tree", "at the desk",
    "in the classroom", "inside the library", "near the computer", "at the station",
    "outside the building", "beside the river", "on the long road", "in the city",
]

ADVERBS = [
    "carefully", "quickly", "slowly", "quietly", "usually", "sometimes", "often",
    "suddenly", "finally", "clearly", "easily", "patiently", "closely", "together",
]

CONNECTORS = [
    "because", "although", "while", "when", "after", "before", "if", "so", "but",
    "and", "yet", "until", "unless",
]

QUESTION_WORDS = ["what", "why", "how", "when", "where", "who"]


def sentence(rng: random.Random, i: int) -> str:
    kind = i % 20
    subject = rng.choice(SUBJECTS)
    verb = rng.choice(VERBS)
    obj = rng.choice(OBJECTS)
    noun = rng.choice(NOUNS)
    adjective = rng.choice(ADJECTIVES)
    place = rng.choice(PLACES)
    adverb = rng.choice(ADVERBS)
    connector = rng.choice(CONNECTORS)

    if kind == 0:
        return f"{subject.capitalize()} {verb} {obj}."
    if kind == 1:
        return f"{subject.capitalize()} {verb} {obj} {place}."
    if kind == 2:
        return f"The {adjective} {noun} becomes easier to understand with practice."
    if kind == 3:
        return f"Learning a {noun} takes time, attention, and repeated practice."
    if kind == 4:
        return f"When {subject} {verb}, it can discover {obj}."
    if kind == 5:
        return f"A useful {noun} can connect one {noun} to another."
    if kind == 6:
        return f"The {adjective} {noun} is followed by a {adjective} {noun}."
    if kind == 7:
        return f"{subject.capitalize()} sees {obj}, pauses, and then {rng.choice(VERBS)} again."
    if kind == 8:
        return f"Small steps can turn a difficult {noun} into a manageable one."
    if kind == 9:
        return f"{subject.capitalize()} {verb} {obj} {adverb}."
    if kind == 10:
        return f"{subject.capitalize()} {verb} {obj} {connector} {subject} needs more practice."
    if kind == 11:
        return f"Although {subject} is {adjective}, it can still {rng.choice(VERBS)} {obj}."
    if kind == 12:
        return f"The {noun} is {adjective}, but the {rng.choice(NOUNS)} is different."
    if kind == 13:
        return f"A person can {rng.choice(VERBS)} a {noun} by using a {adjective} method."
    if kind == 14:
        return f"{rng.choice(QUESTION_WORDS).capitalize()} does {subject} {verb} {obj}?"
    if kind == 15:
        return f"{rng.choice(QUESTION_WORDS).capitalize()} can help a learner understand the {noun}?"
    if kind == 16:
        return f"If {subject} {verb} {obj}, the {noun} may change."
    if kind == 17:
        return f"Before {subject} {verb} {obj}, it checks the {noun}."
    if kind == 18:
        return f"The {subject.removeprefix('the ').removeprefix('a ')} {verb} {obj}, and the {noun} changes as a result."
    return "A good learner asks a question, studies the evidence, and tries another approach."


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
