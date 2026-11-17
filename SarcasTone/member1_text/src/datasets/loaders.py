"""Loaders for the datasets named in the project brief.

Each loader returns examples plus a report of drops and repairs. None of them
invent an audio path or copy a label from one task onto another.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from shared.errors import DatasetSchemaError
from shared.examples import LoadReport, UtteranceExample
from shared.interfaces import parse_sarcasm_label

AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".mp4", ".mkv")
RAVDESS_EMOTIONS = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fear",
    7: "disgust",
    8: "surprise",
}
CREMA_EMOTIONS = {
    "NEU": "neutral",
    "HAP": "happy",
    "SAD": "sad",
    "ANG": "angry",
    "FEA": "fear",
    "DIS": "disgust",
}


def find_audio(example_id: str, directories: list[Path]) -> str | None:
    for directory in directories:
        if not directory.is_dir():
            continue
        for extension in AUDIO_EXTENSIONS:
            candidate = directory / f"{example_id}{extension}"
            if candidate.is_file():
                return str(candidate)
    return None


def _optional_float(value: str) -> float | None:
    text = (value or "").strip()
    if not text:
        return None
    return float(text)


def _blank_to_none(value: str) -> str | None:
    text = (value or "").strip()
    if text.lower() in {"", "none", "nan"}:
        return None
    return text


def load_mustard(path: str | Path, audio_dirs: list[Path] | None = None) -> tuple[list[UtteranceExample], LoadReport]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"MUStARD annotations not found at {source}. Run: python datasets/acquire.py --annotations"
        )
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DatasetSchemaError("MUStARD sarcasm_data.json must be an object keyed by utterance id")
    audio_dirs = audio_dirs or [source.parent / "audio"]
    examples: list[UtteranceExample] = []
    dropped = 0
    warnings: list[str] = []
    for example_id, row in payload.items():
        if not isinstance(row, dict):
            dropped += 1
            warnings.append(f"{example_id}: row is not an object")
            continue
        try:
            text = str(row["utterance"]).strip()
            context = [str(turn) for turn in row.get("context") or []]
            context_speakers = [str(name) for name in row.get("context_speakers") or []]
            label = parse_sarcasm_label(row["sarcasm"])
        except (KeyError, DatasetSchemaError) as exc:
            dropped += 1
            warnings.append(f"{example_id}: {exc}")
            continue
        if not text:
            dropped += 1
            warnings.append(f"{example_id}: empty utterance")
            continue
        if context_speakers and len(context_speakers) != len(context):
            warnings.append(
                f"{example_id}: context_speakers length {len(context_speakers)} != context length {len(context)}"
            )
            context_speakers = []
        examples.append(
            UtteranceExample(
                example_id=str(example_id),
                dataset="mustard",
                text=text,
                context=context,
                context_speakers=context_speakers,
                speaker=str(row.get("speaker") or ""),
                label=label,
                task="sarcasm",
                audio_path=find_audio(str(example_id), audio_dirs),
                show=_blank_to_none(str(row.get("show") or "")),
            )
        )
    report = LoadReport("mustard", str(source), len(examples), dropped, warnings)
    return examples, report


def load_mustard_pp(path: str | Path, audio_dirs: list[Path] | None = None) -> tuple[list[UtteranceExample], LoadReport]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"MUStARD++ CSV not found at {source}. Run: python datasets/acquire.py --annotations"
        )
    audio_dirs = audio_dirs or [source.parent / "audio"]
    with source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"SCENE", "KEY", "SENTENCE", "SPEAKER", "SHOW", "Sarcasm"}
    if not rows or not required.issubset(rows[0]):
        raise DatasetSchemaError(f"MUStARD++ CSV is missing columns: {sorted(required)}")

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        scene = (row.get("SCENE") or "").strip()
        if scene:
            grouped[scene].append(row)

    examples: list[UtteranceExample] = []
    dropped = 0
    warnings: list[str] = []
    for scene, scene_rows in grouped.items():
        context_rows = []
        utterance_rows = []
        for row in scene_rows:
            key = (row.get("KEY") or "").strip()
            if "_c_" in key:
                context_rows.append(row)
            elif key.endswith("_u"):
                utterance_rows.append(row)
            elif (row.get("Sarcasm") or "").strip() in {"0", "1"}:
                utterance_rows.append(row)
                warnings.append(
                    f"Repaired scene {scene}: KEY was {key!r}; treated this row as the target utterance"
                )
        if len(utterance_rows) != 1:
            dropped += 1
            warnings.append(f"{scene}: expected one target utterance, found {len(utterance_rows)}")
            continue
        target = utterance_rows[0]
        try:
            label = parse_sarcasm_label(target["Sarcasm"])
            valence = _optional_float(target.get("Valence") or "")
            arousal = _optional_float(target.get("Arousal") or "")
        except (DatasetSchemaError, ValueError) as exc:
            dropped += 1
            warnings.append(f"{scene}: {exc}")
            continue
        text = (target.get("SENTENCE") or "").replace("\n", " ").strip()
        if not text:
            dropped += 1
            warnings.append(f"{scene}: empty target sentence")
            continue

        def _context_order(row: dict) -> int:
            key = row.get("KEY") or ""
            try:
                return int(key.rsplit("_c_", 1)[1])
            except (IndexError, ValueError):
                return 10**6

        context_rows.sort(key=_context_order)
        context = []
        context_speakers = []
        for row in context_rows:
            sentence = (row.get("SENTENCE") or "").replace("\n", " ").strip()
            if not sentence:
                continue
            context.append(sentence)
            context_speakers.append((row.get("SPEAKER") or "").strip())
        sarcasm_type = _blank_to_none(target.get("Sarcasm_Type") or "")
        if sarcasm_type is not None and sarcasm_type.upper() == "NONE":
            sarcasm_type = None
        examples.append(
            UtteranceExample(
                example_id=scene,
                dataset="mustard++",
                text=text,
                context=context,
                context_speakers=context_speakers,
                speaker=(target.get("SPEAKER") or "").strip(),
                label=label,
                task="sarcasm",
                audio_path=find_audio(scene, audio_dirs),
                show=_blank_to_none(target.get("SHOW") or ""),
                sarcasm_type=sarcasm_type,
                implicit_emotion=_blank_to_none(target.get("Implicit_Emotion") or ""),
                explicit_emotion=_blank_to_none(target.get("Explicit_Emotion") or ""),
                valence=valence,
                arousal=arousal,
            )
        )
    return examples, LoadReport("mustard++", str(source), len(examples), dropped, warnings)


def load_semeval(path: str | Path) -> tuple[list[UtteranceExample], LoadReport]:
    """SemEval-2018 Task 3 task A: tab-separated index, irony label, tweet."""

    source = Path(path)
    examples: list[UtteranceExample] = []
    dropped = 0
    warnings: list[str] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            dropped += 1
            warnings.append(f"line {line_number}: expected 3 tab-separated fields")
            continue
        example_id, label_text, tweet = parts[0], parts[1], "\t".join(parts[2:])
        try:
            label = parse_sarcasm_label(label_text)
        except DatasetSchemaError as exc:
            dropped += 1
            warnings.append(f"line {line_number}: {exc}")
            continue
        examples.append(
            UtteranceExample(
                example_id=example_id.strip(),
                dataset="semeval2018_task3",
                text=tweet.strip(),
                speaker="",
                label=label,
                task="irony",
            )
        )
    return examples, LoadReport("semeval2018_task3", str(source), len(examples), dropped, warnings)


def load_sarc_jsonl(path: str | Path) -> tuple[list[UtteranceExample], LoadReport]:
    source = Path(path)
    examples: list[UtteranceExample] = []
    dropped = 0
    warnings: list[str] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            label = parse_sarcasm_label(row["label"])
            text = str(row["text"]).strip()
        except (json.JSONDecodeError, KeyError, DatasetSchemaError) as exc:
            dropped += 1
            warnings.append(f"line {line_number}: {exc}")
            continue
        if not text:
            dropped += 1
            continue
        context = row.get("context") or []
        if not isinstance(context, list):
            context = [str(context)]
        examples.append(
            UtteranceExample(
                example_id=str(row.get("id") or f"sarc-{line_number}"),
                dataset="sarc",
                text=text,
                context=[str(turn) for turn in context],
                speaker=str(row.get("author") or ""),
                label=label,
                task="sarcasm",
            )
        )
    return examples, LoadReport("sarc", str(source), len(examples), dropped, warnings)


def parse_ravdess_filename(filename: str) -> dict | None:
    """Parse a RAVDESS filename. Song files and malformed names return ``None``."""

    stem = Path(filename).stem
    parts = stem.split("-")
    if len(parts) != 7 or not all(part.isdigit() for part in parts):
        return None
    modality, channel, emotion, intensity, statement, repetition, actor = (int(part) for part in parts)
    if channel != 1 or emotion not in RAVDESS_EMOTIONS:
        return None
    return {
        "emotion": RAVDESS_EMOTIONS[emotion],
        "actor_id": f"{actor:02d}",
        "intensity": intensity,
        "statement": statement,
        "repetition": repetition,
        "modality": modality,
    }


def parse_crema_filename(filename: str) -> dict | None:
    parts = Path(filename).stem.split("_")
    if len(parts) < 4:
        return None
    actor, _sentence, emotion, _level = parts[:4]
    if emotion not in CREMA_EMOTIONS:
        return None
    return {"emotion": CREMA_EMOTIONS[emotion], "actor_id": actor}


def load_emotion_directory(directory: str | Path, dataset: str) -> tuple[list[UtteranceExample], LoadReport]:
    """Index RAVDESS or CREMA-D audio. These examples have no sarcasm label."""

    root = Path(directory)
    if dataset not in {"ravdess", "crema_d"}:
        raise DatasetSchemaError("dataset must be 'ravdess' or 'crema_d'")
    parser = parse_ravdess_filename if dataset == "ravdess" else parse_crema_filename
    examples: list[UtteranceExample] = []
    skipped = 0
    if not root.is_dir():
        raise FileNotFoundError(f"Emotion audio directory not found: {root}")
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in {".wav", ".mp3", ".flac"}:
            continue
        parsed = parser(path.name)
        if parsed is None:
            skipped += 1
            continue
        examples.append(
            UtteranceExample(
                example_id=path.stem,
                dataset=dataset,
                text="",
                speaker=parsed["actor_id"],
                actor_id=parsed["actor_id"],
                emotion=parsed["emotion"],
                label=None,
                task="emotion",
                audio_path=str(path),
            )
        )
    return examples, LoadReport(dataset, str(root), len(examples), skipped, [])
