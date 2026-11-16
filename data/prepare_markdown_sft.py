#!/usr/bin/env python3
"""Prepare CatAI-authored Markdown formatting examples for supervised fine-tuning."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from catai.chat_dataset import default_state, validate_messages

TICK = chr(96)
FENCE = TICK * 3

SCENARIOS: tuple[dict[str, Any], ...] = (
    {
        "title": "a small software release",
        "summary": "A small release is easier to review when testing, packaging, and communication happen in a fixed order.",
        "bullets": ("Run the automated test suite.", "Build the distributable package.", "Review the changelog and version number."),
        "steps": ("Create a release branch from the tested commit.", "Run the full test suite and inspect failures.", "Build and verify the package.", "Tag the verified commit and publish the release notes."),
        "tasks": ("Confirm the target commit is correct.", "Run tests and package checks.", "Review the changelog.", "Publish the tag and release notes."),
        "rows": (("Tests", "Catch regressions before publishing"), ("Package", "Verify that the artifact can be installed"), ("Release notes", "Explain changes to users")),
        "language": "bash",
        "code": "python -m pytest -q\npython -m build",
        "inline": "python -m pytest -q",
        "link_label": "pytest usage guide",
        "url": "https://docs.pytest.org/en/stable/how-to/usage.html",
        "quote": "A release is ready only when the tested commit and published artifact refer to the same source.",
        "next": "Keep the release tag and build output together so the published version can be reproduced.",
    },
    {
        "title": "a Python JSON configuration file",
        "summary": "A configuration file should be easy to parse, explicit about required values, and safe to change.",
        "bullets": ("Use valid JSON syntax.", "Keep keys consistent across environments.", "Validate required values when loading the file."),
        "steps": ("Create the configuration with the required keys.", "Load it with the standard JSON library.", "Check types and required values.", "Report a clear error when validation fails."),
        "tasks": ("Document each configuration key.", "Keep secrets out of committed files.", "Validate the file during startup."),
        "rows": (("Endpoint", "A service URL"), ("Timeout", "A positive number of seconds"), ("Debug", "A boolean feature flag")),
        "language": "python",
        "code": "import json\nfrom pathlib import Path\n\nsettings = json.loads(Path(\"settings.json\").read_text())\nprint(settings[\"timeout\"])",
        "inline": "json.loads()",
        "link_label": "Python JSON documentation",
        "url": "https://docs.python.org/3/library/json.html",
        "quote": "Valid JSON is only the first check; the application must still validate the meaning of each value.",
        "next": "Add a test with a missing key and another with an invalid value.",
    },
    {
        "title": "a reliable HTTP request",
        "summary": "A reliable client sets a timeout, checks the response status, and handles network failures.",
        "bullets": ("Set a finite timeout.", "Check the HTTP status before using the body.", "Handle connection and timeout errors."),
        "steps": ("Build the request with the required headers.", "Send it with a finite timeout.", "Check for non-success status codes.", "Parse the response only after checking it."),
        "tasks": ("Choose a reasonable timeout.", "Handle non-2xx responses.", "Avoid logging credentials or private response data."),
        "rows": (("Timeout", "Prevents an unbounded wait"), ("Status check", "Detects HTTP error responses"), ("Exception handling", "Handles connection failures")),
        "language": "python",
        "code": "import requests\n\nresponse = requests.get(\n    \"https://api.example.com/status\", timeout=10\n)\nresponse.raise_for_status()\nprint(response.json())",
        "inline": "response.raise_for_status()",
        "link_label": "Requests quickstart",
        "url": "https://requests.readthedocs.io/en/latest/user/quickstart/",
        "quote": "Never let a network request wait forever; a timeout is part of the request's correctness.",
        "next": "Test both a successful response and a server that returns an error.",
    },
    {
        "title": "a local project backup",
        "summary": "A backup is useful only when it is complete, stored separately, and possible to restore.",
        "bullets": ("Choose the files that matter.", "Store the archive outside the working directory.", "Test restoration instead of trusting the archive name."),
        "steps": ("Pause writes if the application requires a consistent snapshot.", "Create a compressed archive.", "Copy it to separate storage.", "Restore it into a temporary directory and verify important files."),
        "tasks": ("Create the archive.", "Copy it to separate storage.", "Record the backup date.", "Perform a restore test."),
        "rows": (("Archive", "Collects the selected project files"), ("Off-device copy", "Protects against local disk loss"), ("Restore test", "Checks that the backup is usable")),
        "language": "bash",
        "code": "tar -czf project-backup.tar.gz ./project\ntar -tzf project-backup.tar.gz | head",
        "inline": "tar -czf",
        "link_label": "GNU tar manual",
        "url": "https://www.gnu.org/software/tar/manual/",
        "quote": "A backup that has never been restored is an assumption, not a verified recovery plan.",
        "next": "Keep at least one backup separate from the machine that contains the original project.",
    },
    {
        "title": "a Git feature branch",
        "summary": "A focused feature branch keeps related changes reviewable and makes it easier to revert work safely.",
        "bullets": ("Start from an up-to-date base branch.", "Keep commits focused on one feature.", "Run tests before opening a pull request."),
        "steps": ("Update the base branch.", "Create a clearly named feature branch.", "Implement and test the change.", "Push the branch and open a pull request."),
        "tasks": ("Update the base branch.", "Create the feature branch.", "Commit the completed change.", "Push and request review."),
        "rows": (("Base branch", "The starting point for the work"), ("Feature branch", "Contains the isolated changes"), ("Pull request", "Provides a review and merge record")),
        "language": "bash",
        "code": "git switch main\ngit pull --ff-only\ngit switch -c feature/my-change",
        "inline": "git switch -c",
        "link_label": "Git branch documentation",
        "url": "https://git-scm.com/docs/git-branch",
        "quote": "A branch should make the scope of a change clearer, not hide unrelated work inside one review.",
        "next": "Keep generated files and secrets out of commits unless the project explicitly requires them.",
    },
    {
        "title": "a Python virtual environment",
        "summary": "A virtual environment isolates project dependencies from other Python projects on the same machine.",
        "bullets": ("Create the environment inside the project.", "Activate it before installing packages.", "Record dependencies in the project manifest."),
        "steps": ("Create the environment with the Python interpreter.", "Activate the environment.", "Install the project's declared dependencies.", "Run the project's tests."),
        "tasks": ("Create the environment.", "Activate it.", "Install declared dependencies.", "Run a quick import or test check."),
        "rows": (("Create", "python -m venv .venv"), ("Activate on Linux", "source .venv/bin/activate"), ("Install", "python -m pip install -e .")),
        "language": "bash",
        "code": "python -m venv .venv\nsource .venv/bin/activate\npython -m pip install -e .",
        "inline": "python -m venv",
        "link_label": "Python venv documentation",
        "url": "https://docs.python.org/3/library/venv.html",
        "quote": "Keep each project's dependencies isolated so one project's upgrade does not unexpectedly break another.",
        "next": "Do not commit the virtual environment directory; recreate it from the dependency manifest.",
    },
    {
        "title": "a CSV data-cleaning task",
        "summary": "A safe data-cleaning script preserves the source file and makes each transformation explicit.",
        "bullets": ("Read the input with the correct encoding.", "Normalize only fields with a defined rule.", "Write the result to a new output file."),
        "steps": ("Inspect a few input rows.", "Define the columns and cleanup rules.", "Transform the records.", "Write a new file and compare row counts."),
        "tasks": ("Keep the original input unchanged.", "Check for missing values.", "Normalize whitespace where appropriate.", "Compare input and output row counts."),
        "rows": (("Input", "Original source file"), ("Transform", "Documented cleanup rules"), ("Output", "New cleaned file")),
        "language": "python",
        "code": "import csv\n\nwith open(\"input.csv\", newline=\"\", encoding=\"utf-8\") as source:\n    rows = list(csv.DictReader(source))\n\nprint(f\"Loaded {len(rows)} rows\")",
        "inline": "csv.DictReader",
        "link_label": "Python CSV documentation",
        "url": "https://docs.python.org/3/library/csv.html",
        "quote": "Never overwrite the only copy of the raw data during a cleaning step.",
        "next": "Add a small fixture with an empty field and a quoted comma to test the parser.",
    },
    {
        "title": "a responsive CSS layout",
        "summary": "A responsive layout adapts to available space without forcing content outside the viewport.",
        "bullets": ("Use flexible widths instead of fixed page-wide dimensions.", "Allow grid columns to wrap.", "Check the layout at narrow and wide viewport sizes."),
        "steps": ("Start with the narrow-screen layout.", "Add flexible grid or flex rules.", "Introduce breakpoints only when the content needs them.", "Test long text and large controls."),
        "tasks": ("Avoid fixed widths for main content.", "Check a narrow viewport.", "Check a wide viewport.", "Test long labels and overflowing code."),
        "rows": (("Grid", "Useful for two-dimensional layouts"), ("Flexbox", "Useful for one-dimensional alignment"), ("Media query", "Applies styles at selected viewport conditions")),
        "language": "css",
        "code": ".layout {\n  display: grid;\n  grid-template-columns: repeat(auto-fit, minmax(16rem, 1fr));\n  gap: 1rem;\n}",
        "inline": "grid-template-columns",
        "link_label": "MDN CSS display reference",
        "url": "https://developer.mozilla.org/en-US/docs/Web/CSS/display",
        "quote": "Responsive design should preserve readable content, not merely shrink every element.",
        "next": "Test keyboard focus and content overflow as well as the visual appearance.",
    },
)

FORMATS = (
    "headings", "bullets", "numbered_steps", "table", "checklist",
    "fenced_code", "inline_link", "blockquote", "mixed_formatting",
)


def _render(scenario: dict[str, Any], format_name: str) -> tuple[str, str]:
    title = scenario["title"].capitalize()
    summary = scenario["summary"]
    items = scenario["bullets"]
    steps = scenario["steps"]

    if format_name == "headings":
        prompt = f"Write a concise Markdown guide about {scenario['title']}. Use headings and short sections."
        answer = (
            f"# {title}\n\n{summary}\n\n## Key points\n\n"
            + "\n".join(f"- {item}" for item in items)
            + f"\n\n## Next step\n\n{scenario['next']}"
        )
    elif format_name == "bullets":
        prompt = f"Summarize {scenario['title']} as a scannable Markdown bullet list with one nested detail."
        answer = (
            f"Here are the main points for {scenario['title']}:\n\n"
            + "\n".join(f"- **{item.split('.')[0]}:** {item[0].lower() + item[1:]}\n  - Keep the rule consistent throughout the task." for item in items)
            + f"\n\n**Remember:** {scenario['next']}"
        )
    elif format_name == "numbered_steps":
        prompt = f"Give me a numbered Markdown procedure for {scenario['title']}."
        answer = (
            "## Steps\n\n"
            + "\n".join(f"{index}. {step}" for index, step in enumerate(steps, 1))
            + f"\n\nAfterward, {scenario['next'][0].lower() + scenario['next'][1:]}"
        )
    elif format_name == "table":
        prompt = f"Organize the important parts of {scenario['title']} in a Markdown table."
        answer = (
            "Use this quick reference:\n\n"
            "| Part | Purpose |\n| --- | --- |\n"
            + "\n".join(f"| {part} | {purpose} |" for part, purpose in scenario["rows"])
            + f"\n\n{scenario['next']}"
        )
    elif format_name == "checklist":
        prompt = f"Turn {scenario['title']} into a practical Markdown task checklist."
        answer = (
            "## Checklist\n\n"
            + "\n".join(f"- [ ] {task}" for task in scenario["tasks"])
            + "\n\nMark each item complete only after checking the result."
        )
    elif format_name == "fenced_code":
        prompt = f"Show a short example for {scenario['title']} with a fenced {scenario['language']} code block."
        answer = (
            "Here is a minimal example:\n\n"
            + f"{FENCE}{scenario['language']}\n{scenario['code']}\n{FENCE}\n\n"
            + f"The important command or construct is {TICK}{scenario['inline']}{TICK}. "
            + scenario["next"]
        )
    elif format_name == "inline_link":
        prompt = f"Explain {scenario['title']} with inline code and a useful Markdown documentation link."
        answer = (
            f"{summary}\n\n"
            + f"Use {TICK}{scenario['inline']}{TICK} where it fits the task. "
            + f"See the [{scenario['link_label']}]({scenario['url']}) for the reference details.\n\n"
            + scenario["next"]
        )
    elif format_name == "blockquote":
        prompt = f"Explain {scenario['title']} briefly and highlight the most important caution in a Markdown blockquote."
        answer = (
            f"{summary}\n\n"
            + f"> **Important:** {scenario['quote']}\n\n"
            + scenario["next"]
        )
    elif format_name == "mixed_formatting":
        prompt = f"Write a compact Markdown note about {scenario['title']} using a heading, bold text, bullets, and a horizontal rule."
        answer = (
            f"## {title}\n\n**Goal:** {summary}\n\n"
            + "\n".join(f"- {item}" for item in items)
            + f"\n\n---\n\n**Next:** {scenario['next']}"
        )
    else:
        raise ValueError(f"unknown Markdown format: {format_name}")

    messages = validate_messages([
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer},
    ])
    return prompt, messages[-1]["content"]


def build_examples() -> list[dict[str, Any]]:
    """Build the complete deterministic, CatAI-authored Markdown SFT sample."""
    records: list[dict[str, Any]] = []
    for format_name in FORMATS:
        for scenario_index, scenario in enumerate(SCENARIOS):
            prompt, answer = _render(scenario, format_name)
            records.append({
                "state": default_state(),
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer},
                ],
                "dataset_source": "catai_markdown_sft",
                "dataset_license": "Apache-2.0",
                "dataset_repository": "bananakitssu/catai",
                "dataset_version": 1,
                "dataset_category": format_name,
                "dataset_record_id": f"{format_name}-{scenario_index + 1:02d}",
                "dataset_attribution": "Original CatAI-authored Markdown formatting example.",
            })
    return records


def _write_jsonl(records: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".part")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def prepare(output: Path, *, max_examples: int = 72, seed: int = 42) -> int:
    """Write a deterministic shuffled sample of original Markdown examples."""
    if max_examples < 1:
        raise ValueError("--max-examples must be positive")
    if seed < 0:
        raise ValueError("--seed must be non-negative")
    records = build_examples()
    if max_examples > len(records):
        raise ValueError(f"--max-examples cannot exceed the {len(records)} authored examples")
    random.Random(seed).shuffle(records)
    selected = records[:max_examples]
    _write_jsonl(selected, output)
    print(f"Prepared {len(selected):,} CatAI-authored Markdown examples at {output}")
    return len(selected)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare CatAI-authored Markdown-formatting SFT examples.")
    parser.add_argument("--output", type=Path, default=Path("data/markdown_sft_chat.jsonl"))
    parser.add_argument("--max-examples", type=int, default=72)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    prepare(args.output, max_examples=args.max_examples, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
