"""MUStARD++ loader.

The CSV is one row per turn. Context keys end in ``_c_<index>``. The target
key ends in ``_u`` and carries the labels. This loader reads a local file and
does not scrape Google Drive.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from SarcasTone.member1_text.member1_text.src.member1_text.canonicalize import CanonicalExample, require_local_file
from SarcasTone.member1_text.member1_text.src.member1_text.config import load_config, resolve_repo_path

SOURCE_URL = "https://github.com/cfiltnlp/MUStARD_Plus_Plus"
CITATION = (
    "Anupama Ray, Shubham Mishra, Apoorva Nunna, and Pushpak Bhattacharyya. "
    "A Multimodal Corpus for Emotion Recognition in Sarcasm. LREC 2022."
)
REQUIRED_COLUMNS = (
    "SCENE",
    "KEY",
    "SENTENCE",
    "END_TIME",
    "SPEAKER",
    "SHOW",
    "Sarcasm",
    "Sarcasm_Type",
    "Implicit_Emotion",
    "Explicit_Emotion",
    "Valence",
    "Arousal",
)
# Short codes named by the paper's four sarcastic types plus non-sarcastic.
# A code outside this map fails. Meanings are not guessed.
SARCASM_TYPE_MEANINGS = {
    "NONE": "non-sarcastic",
    "PRO": "propositional",
    "EMB": "embedded",
    "LIKE": "like-prefixed",
    "ILLOC": "illocutionary",
}
_CONTEXT_KEY = re.compile(r"_c_(\d+)$")


def default_text_path() -> Path:
    return resolve_repo_path(load_config()["datasets"]["mustard_pp"]["local_text_path"])


def load_mustard_pp(
    path: Path | None = None,
    *,
    task: str = "sarcasm",
    split: str = "all",
) -> list[CanonicalExample]:
    """Group turns into one example per scene. Unknown sarcasm codes fail."""
    csv_path = require_local_file("MUStARD++", path or default_text_path(), SOURCE_URL)
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"MUStARD++ file {csv_path} has no header.")
        missing = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"MUStARD++ file is missing columns: {missing}")
        rows = list(reader)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        scene = (row.get("SCENE") or "").strip()
        if not scene:
            raise ValueError("MUStARD++ row is missing SCENE.")
        grouped[scene].append(row)
    examples = [_scene_to_example(scene, scene_rows, task=task, split=split) for scene, scene_rows in grouped.items()]
    if not examples:
        raise ValueError(f"MUStARD++ file {csv_path} produced no scenes.")
    return examples


def _scene_to_example(
    scene: str,
    rows: list[dict[str, str]],
    *,
    task: str,
    split: str,
) -> CanonicalExample:
    context_rows: list[tuple[int, dict[str, str]]] = []
    targets: list[dict[str, str]] = []
    for row in rows:
        key = (row.get("KEY") or "").strip()
        kind, index = _classify_key(key)
        if kind == "context":
            context_rows.append((index if index is not None else 0, row))
        else:
            targets.append(row)
    if len(targets) != 1:
        raise ValueError(f"MUStARD++ scene {scene} must have exactly one target row, found {len(targets)}.")
    target = targets[0]
    sentence = (target.get("SENTENCE") or "").strip()
    if not sentence:
        raise ValueError(f"MUStARD++ scene {scene} target row has an empty SENTENCE.")
    context_rows.sort(key=lambda item: item[0])
    turns = [
        {"speaker": (row.get("SPEAKER") or "").strip(), "text": (row.get("SENTENCE") or "").strip()}
        for _, row in context_rows
    ]
    sarcasm = _parse_binary(target.get("Sarcasm", ""), scene)
    sarcasm_type = (target.get("Sarcasm_Type") or "").strip()
    if sarcasm_type not in SARCASM_TYPE_MEANINGS:
        raise ValueError(
            f"MUStARD++ scene {scene} has unknown Sarcasm_Type {sarcasm_type!r}. "
            f"Known codes: {sorted(SARCASM_TYPE_MEANINGS)}. The loader does not guess a meaning."
        )
    explicit = (target.get("Explicit_Emotion") or "").strip()
    implicit = (target.get("Implicit_Emotion") or "").strip()
    valence = _parse_scale(target.get("Valence", ""), scene, "Valence")
    arousal = _parse_scale(target.get("Arousal", ""), scene, "Arousal")
    label, label_meaning = _task_label(
        task,
        scene=scene,
        sarcasm=sarcasm,
        explicit=explicit,
        valence=valence,
        arousal=arousal,
    )
    present_codes = {sarcasm_type: SARCASM_TYPE_MEANINGS[sarcasm_type]}
    return CanonicalExample(
        example_id=(target.get("KEY") or "").strip() or f"{scene}_u",
        dataset="mustard_pp",
        split=split,
        text=sentence,
        context_turns=turns,
        text_source="gold",
        label=label,
        label_meaning=label_meaning,
        extra={
            "scene": scene,
            "speaker": (target.get("SPEAKER") or "").strip(),
            "show": (target.get("SHOW") or "").strip(),
            "sarcasm": sarcasm,
            "sarcasm_type": sarcasm_type,
            "sarcasm_type_meaning": SARCASM_TYPE_MEANINGS[sarcasm_type],
            "sarcasm_type_codes_present": present_codes,
            "implicit_emotion": implicit,
            "explicit_emotion": explicit,
            "valence": valence,
            "arousal": arousal,
            "overlap_text": sentence,
        },
        overlap_key=scene,
    )


def _classify_key(key: str) -> tuple[str, int | None]:
    if key.endswith("_u"):
        return "target", None
    match = _CONTEXT_KEY.search(key)
    if match:
        return "context", int(match.group(1))
    raise ValueError(f"MUStARD++ KEY {key!r} is neither a context row nor a target row.")


def _parse_binary(value: str, scene: str) -> int:
    text = (value or "").strip()
    if text not in {"0", "1"}:
        raise ValueError(f"MUStARD++ scene {scene} Sarcasm label must be 0 or 1, got {value!r}.")
    return int(text)


def _parse_scale(value: str, scene: str, field: str) -> float:
    text = (value or "").strip()
    try:
        number = float(text)
    except ValueError as exc:
        raise ValueError(f"MUStARD++ scene {scene} {field} must be numeric, got {value!r}.") from exc
    if not 1.0 <= number <= 9.0:
        raise ValueError(f"MUStARD++ scene {scene} {field} must be from 1 to 9, got {number}.")
    return number


def _task_label(
    task: str,
    *,
    scene: str,
    sarcasm: int,
    explicit: str,
    valence: float,
    arousal: float,
) -> tuple[int | float | str, str]:
    if task == "sarcasm":
        return sarcasm, "sarcasm"
    if task == "explicit_emotion":
        if not explicit:
            raise ValueError(f"MUStARD++ scene {scene} is missing Explicit_Emotion.")
        return explicit, "explicit_emotion"
    if task == "valence":
        return valence, "valence"
    if task == "arousal":
        return arousal, "arousal"
    raise ValueError(f"Unsupported MUStARD++ task {task!r}.")


def observed_sarcasm_types(examples: list[CanonicalExample]) -> dict[str, str]:
    """Codes actually present. Meanings come only from the documented map."""
    observed: dict[str, str] = {}
    for example in examples:
        code = str(example.extra.get("sarcasm_type", ""))
        if code and code not in SARCASM_TYPE_MEANINGS:
            raise ValueError(f"Unknown sarcasm type code {code!r}.")
        if code:
            observed[code] = SARCASM_TYPE_MEANINGS[code]
    return observed
