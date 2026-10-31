"""Prepare emotion-aware supervised examples from public emotion datasets."""
from __future__ import annotations
from collections import defaultdict
from typing import Any, Iterable
from .chat_dataset import default_state

GO_EMOTIONS_LABELS = (
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
)
DAILY_DIALOG_EMOTIONS = (
    "no emotion", "anger", "disgust", "fear", "happiness", "sadness", "surprise",
)
DATASET_LICENSES = {
    "go_emotions": "apache-2.0",
    "daily_dialog": "CC-BY-NC-SA-4.0",
    "empathetic_dialogues": "CC-BY-NC-4.0",
}
EMOTION_RESPONSE_GUIDANCE = (
    "Emotion-aware conversation training. The emotion annotation is a useful cue, "
    "not certainty about what the person feels. Pay attention to the user's actual "
    "words and context. Respond naturally and proportionately. Acknowledge feelings "
    "without claiming you literally share them or understand exactly. Do not use "
    "formulaic claims such as 'I feel your frustration'. Offer practical help when "
    "it fits, and ask a gentle question when the person seems to want to talk."
)
EMOTION_RECOGNITION_GUIDANCE = (
    "Emotion-recognition exercise. Identify only the emotions supported by the "
    "message. A message can express more than one emotion, or none clearly. Do not "
    "diagnose the writer or invent hidden motives. Return the annotated emotion "
    "labels only, in a concise comma-separated list."
)

def _clean(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.replace("_comma_", ",").split())

def _record(source: str, system_instruction: str, messages: list[dict[str, str]],
            emotion_labels: list[str]) -> dict[str, Any]:
    return {
        "state": default_state(),
        "messages": [{"role": "system", "content": system_instruction}, *messages],
        "dataset_source": source,
        "dataset_license": DATASET_LICENSES[source],
        "emotion_labels": emotion_labels,
    }

def go_emotions_records(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert GoEmotions simplified rows into grounded emotion-label tasks."""
    records: list[dict[str, Any]] = []
    for row in rows:
        text = _clean(row.get("text"))
        if not text:
            continue
        raw_labels = row.get("labels", [])
        if not isinstance(raw_labels, (list, tuple)):
            raise ValueError("GoEmotions labels must be a list of label IDs")
        labels: list[str] = []
        for raw_label in raw_labels:
            if isinstance(raw_label, bool) or not isinstance(raw_label, int):
                raise ValueError(f"invalid GoEmotions label ID: {raw_label!r}")
            if raw_label < 0 or raw_label >= len(GO_EMOTIONS_LABELS):
                raise ValueError(f"unknown GoEmotions label ID: {raw_label}")
            label = GO_EMOTIONS_LABELS[raw_label]
            if label not in labels:
                labels.append(label)
        if not labels:
            continue
        records.append(_record(
            "go_emotions", EMOTION_RECOGNITION_GUIDANCE,
            [
                {"role": "user", "content": "Identify the emotions expressed in this message:\n" + text},
                {"role": "assistant", "content": "Emotions: " + ", ".join(labels)},
            ],
            labels,
        ))
    return records

def daily_dialog_records(dialogues: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert DailyDialog's paired turns into emotion-conditioned responses."""
    records: list[dict[str, Any]] = []
    for dialogue_index, row in enumerate(dialogues, start=1):
        turns, emotions = row.get("dialog"), row.get("emotion")
        if not isinstance(turns, (list, tuple)) or not isinstance(emotions, (list, tuple)):
            raise ValueError(f"DailyDialog row {dialogue_index} needs dialog and emotion arrays")
        if len(turns) != len(emotions):
            raise ValueError(
                f"DailyDialog row {dialogue_index} has {len(turns)} turns but "
                f"{len(emotions)} emotion labels"
            )
        for user_index in range(0, len(turns) - 1, 2):
            user_text, reply_text = _clean(turns[user_index]), _clean(turns[user_index + 1])
            if not user_text or not reply_text:
                continue
            raw_emotion = emotions[user_index]
            if isinstance(raw_emotion, int) and not isinstance(raw_emotion, bool):
                if raw_emotion < 0 or raw_emotion >= len(DAILY_DIALOG_EMOTIONS):
                    raise ValueError(f"unknown DailyDialog emotion ID: {raw_emotion}")
                emotion = DAILY_DIALOG_EMOTIONS[raw_emotion]
            else:
                emotion = _clean(raw_emotion).lower()
                if emotion not in DAILY_DIALOG_EMOTIONS:
                    raise ValueError(f"unknown DailyDialog emotion: {raw_emotion!r}")
            label = "neutral" if emotion == "no emotion" else emotion
            instruction = EMOTION_RESPONSE_GUIDANCE + f"\nThe user's latest turn is annotated as: {label}."
            records.append(_record(
                "daily_dialog", instruction,
                [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": reply_text},
                ],
                [] if label == "neutral" else [label],
            ))
    return records

def empathetic_dialogues_records(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert EmpatheticDialogues listener replies into emotional conversations."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row_number, row in enumerate(rows, start=1):
        conv_id, utterance = _clean(row.get("conv_id")), _clean(row.get("utterance"))
        if not conv_id or not utterance:
            continue
        try:
            utterance_index, speaker_index = int(row["utterance_idx"]), int(row["speaker_idx"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"EmpatheticDialogues row {row_number} has invalid turn/speaker metadata"
            ) from exc
        grouped[conv_id].append({
            "utterance_idx": utterance_index, "speaker_idx": speaker_index,
            "utterance": utterance, "context": _clean(row.get("context")).lower(),
            "prompt": _clean(row.get("prompt")),
        })

    records: list[dict[str, Any]] = []
    for conv_id in sorted(grouped):
        conversation = sorted(grouped[conv_id], key=lambda item: item["utterance_idx"])
        if len(conversation) < 2:
            continue
        first_speaker = conversation[0]["speaker_idx"]
        for target_index in range(1, len(conversation)):
            previous, target = conversation[target_index - 1], conversation[target_index]
            previous_role = "user" if previous["speaker_idx"] == first_speaker else "assistant"
            target_role = "user" if target["speaker_idx"] == first_speaker else "assistant"
            if previous_role != "user" or target_role != "assistant":
                continue
            start_index = max(0, target_index - 3)
            first_role = (
                "user" if conversation[start_index]["speaker_idx"] == first_speaker else "assistant"
            )
            if first_role != "user":
                start_index += 1
            recent = conversation[start_index:target_index + 1]
            messages = [
                {
                    "role": "user" if turn["speaker_idx"] == first_speaker else "assistant",
                    "content": turn["utterance"],
                }
                for turn in recent
            ]
            if not messages or messages[0]["role"] != "user" or messages[-1]["role"] != "assistant":
                continue
            emotion = previous["context"] or conversation[0]["context"] or "unspecified"
            prompt = previous["prompt"] or conversation[0]["prompt"]
            instruction = (
                EMOTION_RESPONSE_GUIDANCE
                + f"\nThe conversation's annotated emotional context is: {emotion}. "
                "Treat this as background context, not proof of the user's current mood."
            )
            if prompt:
                instruction += "\nSituation shared in the original prompt: " + prompt
            records.append(_record(
                "empathetic_dialogues", instruction, messages,
                [emotion] if emotion != "unspecified" else [],
            ))
    return records

def write_jsonl(path: str, records: Iterable[dict[str, Any]]) -> int:
    """Write normalized records as UTF-8 JSONL and return the number written."""
    import json
    from pathlib import Path
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    count = 0
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
                stream.write("\n")
                count += 1
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return count
