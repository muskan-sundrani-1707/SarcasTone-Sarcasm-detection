"""SemEval-2018 Task 3 loader.

Parsing is first token, second token, and the remainder of the line.
Task A stays binary irony. Task B stays four-class. Irony is not renamed
to sarcasm. Files whose names contain ``ironyHashtags`` are rejected.
"""

from __future__ import annotations

from pathlib import Path

from member1_text.canonicalize import CanonicalExample, require_local_file
from member1_text.config import load_config, resolve_repo_path

SOURCE_URL = "https://github.com/Cyvhee/SemEval2018-Task3"
LICENSE_NOTE = (
    "CC BY-NC-SA 4.0. Academic use. Tweet text is not redistributed and the "
    "task repository is not cloned by this package."
)
CITATION = (
    "Cynthia Van Hee, Els Lefever, and Veronique Hoste. "
    "SemEval-2018 Task 3: Irony Detection in English Tweets. SemEval 2018."
)
TASK_A_LABELS = {"0": 0, "1": 1}
TASK_B_LABELS = {"0": 0, "1": 1, "2": 2, "3": 3}
TASK_B_MEANINGS = {
    0: "non-irony",
    1: "verbal irony with polarity contrast",
    2: "other verbal irony",
    3: "situational irony",
}


def local_dir() -> Path:
    return resolve_repo_path(load_config()["datasets"]["semeval2018"]["local_dir"])


def load_semeval(
    path: Path,
    *,
    task: str,
    split: str,
    allow_hashtag_leak: bool = False,
) -> list[CanonicalExample]:
    """Parse one SemEval file. The returned count is the count actually read."""
    if task not in {"taskA", "taskB"}:
        raise ValueError("SemEval task must be taskA or taskB.")
    if "ironyHashtags" in Path(path).name and not allow_hashtag_leak:
        raise ValueError(
            f"Refusing {Path(path).name} because irony hashtags leak the label. "
            "Pass allow_hashtag_leak=True only for an explicit ablation."
        )
    text_path = require_local_file("SemEval-2018 Task 3", path, SOURCE_URL)
    label_map = TASK_A_LABELS if task == "taskA" else TASK_B_LABELS
    label_meaning = "irony" if task == "taskA" else "irony_type"
    examples: list[CanonicalExample] = []
    with text_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.rstrip("\n")
            if not stripped.strip():
                continue
            if line_number == 1 and stripped.lower().startswith("tweet index"):
                continue
            example_id, label_token, text = _parse_line(stripped, line_number)
            if label_token not in label_map:
                raise ValueError(
                    f"SemEval {task} line {line_number} has label {label_token!r}, "
                    f"which is outside {sorted(label_map)}."
                )
            if not text.strip():
                raise ValueError(f"SemEval line {line_number} has empty tweet text.")
            label = label_map[label_token]
            extra = {"overlap_text": text}
            if task == "taskB":
                extra["irony_type_meaning"] = TASK_B_MEANINGS[label]
            examples.append(
                CanonicalExample(
                    example_id=example_id,
                    dataset="semeval2018",
                    split=split,
                    text=text,
                    context_turns=[],
                    text_source="gold",
                    label=label,
                    label_meaning=label_meaning,
                    extra=extra,
                    overlap_key=example_id,
                )
            )
    if not examples:
        raise ValueError(f"SemEval file {text_path} produced no tweets.")
    return examples


def _parse_line(line: str, line_number: int) -> tuple[str, str, str]:
    parts = line.split(maxsplit=2)
    if len(parts) < 3:
        raise ValueError(
            f"SemEval line {line_number} must contain an index, a label, and tweet text."
        )
    return parts[0], parts[1], parts[2]


def default_train_path(task: str) -> Path:
    config = load_config()["datasets"]["semeval2018"]
    name = config["task_a_train"] if task == "taskA" else config["task_b_train"]
    return local_dir() / name


def load_official_test(task: str, path: Path | None = None) -> list[CanonicalExample]:
    """Load a gold test file only when that file is already local."""
    candidate = path or (local_dir() / f"SemEval2018-T3-test-{task}.txt")
    return load_semeval(candidate, task=task, split="test")
