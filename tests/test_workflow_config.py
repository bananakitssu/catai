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


def test_hosted_base_training_is_a_32_stage_recovery_chain() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
    blocks = _base_training_blocks(workflow)

    expected_names = ["base_training"] + [
        f"base_training{index}" for index in range(2, 33)
    ]
    assert [name for name, _ in blocks] == expected_names

    for index, (name, block) in enumerate(blocks, start=1):
        assert "--vocab-size 2048" in block
        assert "--warmup-fraction 0.03" in block
        assert "--minimum-learning-rate-ratio 0.1" in block

        if index == 1:
            previous_artifact = None
        else:
            previous_artifact = (
                "catai-base-recovery"
                if index == 2
                else f"catai-base-recovery{index - 1}"
            )
            assert f"name: {previous_artifact}" in block

        if index < 32:
            assert "--max-batches-per-run-fraction 0.03125" in block
            output_artifact = (
                "catai-base-recovery"
                if index == 1
                else f"catai-base-recovery{index}"
            )
            assert f"name: {output_artifact}" in block
        else:
            assert "needs: [test, base_training31]" in block
            assert "scripts/evaluate_base.py" in block
            assert "name: catai-base-checkpoint" in block


def test_hosted_workflow_defaults_to_small_and_trains_on_approved_expansion() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    model_options = re.search(
        r"      model_size:\n(?P<body>.*?)(?=\n      publish_model:)",
        workflow,
        re.DOTALL,
    )
    assert model_options is not None
    assert "default: small" in workflow
    assert "description: \"Hosted CI supports tiny/small." in model_options.group("body")
    assert re.search(r"options:\n\s+- tiny\n\s+- small", model_options.group("body"))
    assert "- base" not in model_options.group("body")
    assert "- medium" not in model_options.group("body")

    assert "data/expanded_sft_chat.jsonl" in workflow
    assert workflow.count("--expanded-chat-dataset data/expanded_sft_chat.jsonl") == 2
    assert workflow.count("--expanded-chat-ratio 0.15") == 2
    assert workflow.count("needs.dataset_expansion.result == 'success'") == 2
