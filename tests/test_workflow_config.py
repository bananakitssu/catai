from __future__ import annotations

import re
from pathlib import Path


WORKFLOW_PATH = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"


def _base_training_blocks(workflow: str) -> list[tuple[str, str]]:
    matches = list(re.finditer(r"^  (base_training\d*):\n", workflow, re.MULTILINE))
    blocks = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else workflow.find(
            "  dataset_expansion:\n", match.end()
        )
        if end < 0:
            end = len(workflow)
        blocks.append((match.group(1), workflow[match.start():end]))
    return blocks


def test_hosted_base_training_is_a_128_stage_recovery_chain() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
    blocks = _base_training_blocks(workflow)

    expected_names = ["base_training"] + [
        f"base_training{index}" for index in range(2, 129)
    ]
    assert [name for name, _ in blocks] == expected_names

    for index, (name, block) in enumerate(blocks, start=1):
        assert "--vocab-size 2048" in block
        assert "--warmup-fraction 0.03" in block
        assert "--minimum-learning-rate-ratio 0.1" in block
        assert '--epochs "$BASE_EPOCHS"' in block
        assert '--batch-size "$BASE_BATCH_SIZE"' in block
        assert '--sequence-length "$SEQUENCE_LENGTH"' in block
        assert '--max-tokens "$MAX_TOKENS"' in block
        assert "--minimum-tokens-per-parameter 20" in block
        assert "timeout-minutes: 360" in block

        if index == 1:
            assert "--max-batches-per-run-fraction 0.0078125" in block
        else:
            previous_artifact = (
                "catai-base-recovery"
                if index == 2
                else f"catai-base-recovery{index - 1}"
            )
            assert f"name: {previous_artifact}" in block

        if index < 128:
            assert "--max-batches-per-run-fraction 0.0078125" in block
            output_artifact = (
                "catai-base-recovery"
                if index == 1
                else f"catai-base-recovery{index}"
            )
            assert f"name: {output_artifact}" in block
        else:
            assert "needs: [test, base_training127]" in block
            assert "--max-batches-per-run-fraction" not in block
            assert "scripts/evaluate_base.py" in block
            assert "name: catai-base-checkpoint" in block

    assert workflow.count("needs.base_training128.result == 'success'") == 2
    assert workflow.count("needs: [test, base_training128,") == 2


def test_hosted_workflow_defaults_to_base_and_trains_on_approved_expansion() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    model_options = re.search(
        r"      model_size:\n(?P<body>.*?)(?=\n      publish_model:)",
        workflow,
        re.DOTALL,
    )
    assert model_options is not None
    body = model_options.group("body")
    assert "default: base" in body
    assert 'description: "Hosted CI supports tiny, small, and base.' in body
    assert re.search(r"options:\n\s+- tiny\n\s+- small\n\s+- base", body)
    assert "- medium" not in body

    assert "data/expanded_sft_chat.jsonl" in workflow
    assert workflow.count("--expanded-chat-dataset data/expanded_sft_chat.jsonl") == 2
    assert workflow.count("--expanded-chat-ratio 0.15") == 2
    assert workflow.count("--markdown-dataset data/markdown_sft_chat.jsonl") == 2
    assert workflow.count("--markdown-ratio 0.05") == 2
    assert "data/prepare_markdown_sft.py" in workflow
    assert workflow.count("needs.dataset_expansion.result == 'success'") == 2


def test_hosted_base_corpus_budget_is_expanded_and_cache_is_invalidated() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
    blocks = _base_training_blocks(workflow)
    expected_arguments = (
        "--target-chars 1400000000",
        "--fineweb-edu-chars 210000000",
        "--cosmopedia-chars 140000000",
        "--common-corpus-chars 140000000",
        "--common-corpus-max-documents 7000",
    )

    for name, block in blocks:
        for argument in expected_arguments:
            assert argument in block, f"{name} is missing {argument}"

    expansion = re.search(
        r"^  corpus_expansion:\n(?P<body>.*?)(?=^  [a-zA-Z0-9_]+:\n|\Z)",
        workflow,
        re.MULTILINE | re.DOTALL,
    )
    assert expansion is not None
    for argument in expected_arguments:
        assert argument in expansion.group("body")

    cache_key = "catai-corpus-curated-v6-${{ hashFiles('data/download_corpus.py') }}"
    assert "catai-corpus-curated-v4-" not in workflow
    assert workflow.count(f"key: {cache_key}") >= len(blocks) + 1
