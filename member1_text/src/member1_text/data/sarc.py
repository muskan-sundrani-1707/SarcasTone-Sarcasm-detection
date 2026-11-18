"""SARC 2.0 balanced loader.

Rows are ``ancestors | responses | labels``. Response ids are looked up in
``comments.json``. The official train and test files are not reshuffled.
"""

from __future__ import annotations

import bz2
import json
from pathlib import Path
from typing import Any

from member1_text.canonicalize import CanonicalExample, require_local_file
from member1_text.config import load_config, resolve_repo_path

SOURCE_URL = "https://nlp.cs.princeton.edu/old/SARC/2.0/"
CITATION = (
    "Mikhail Khodak, Nikunj Saunshi, and Kiran Vodrahalli. "
    "A Large Self-Annotated Corpus for Sarcasm. LREC 2018."
)
LICENSE_NOTE = (
    "Citation required. The corpus readme does not state an SPDX license. "
    "The GitHub evaluation code license does not cover the corpus."
)


def _config() -> dict[str, Any]:
    return load_config()["datasets"]["sarc"]


def local_dir() -> Path:
    return resolve_repo_path(_config()["local_dir"])


def _open_text(path: Path):
    if path.suffix == ".bz2":
        return bz2.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def load_comments(path: Path | None = None) -> dict[str, dict[str, Any]]:
    comments_name = _config()["comments_file"]
    comments_path = require_local_file(
        "SARC 2.0",
        path or (local_dir() / comments_name),
        SOURCE_URL,
    )
    with _open_text(comments_path) as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("SARC comments.json must be an object keyed by comment id.")
    comments: dict[str, dict[str, Any]] = {}
    for comment_id, value in payload.items():
        comments[str(comment_id)] = _comment_record(str(comment_id), value)
    return comments


def _comment_record(comment_id: str, value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        return {"text": value}
    if isinstance(value, dict):
        if "text" not in value:
            raise ValueError(f"SARC comment {comment_id} has no text field.")
        return value
    raise ValueError(f"SARC comment {comment_id} has unsupported type {type(value).__name__}.")


def load_sarc(
    directory: Path | None = None,
    *,
    split: str = "train",
    comments: dict[str, dict[str, Any]] | None = None,
) -> list[CanonicalExample]:
    """One example per response id. ``pol/`` and unbalanced files are not read."""
    config = _config()
    root = directory or local_dir()
    if split not in {"train", "test"}:
        raise ValueError("SARC split must be 'train' or 'test'. Official files are not reshuffled.")
    filename = config["train_file"] if split == "train" else config["test_file"]
    table_path = require_local_file("SARC 2.0", root / filename, SOURCE_URL)
    comment_table = comments if comments is not None else load_comments(root / config["comments_file"])
    examples: list[CanonicalExample] = []
    with _open_text(table_path) as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            examples.extend(_parse_line(stripped, line_number, comment_table, split=split))
    if not examples:
        raise ValueError(f"SARC file {table_path} produced no responses.")
    return examples


def _parse_line(
    line: str,
    line_number: int,
    comments: dict[str, dict[str, Any]],
    *,
    split: str,
) -> list[CanonicalExample]:
    parts = [part.strip() for part in line.split("|")]
    if len(parts) != 3:
        raise ValueError(f"SARC line {line_number} must have three pipe-separated fields.")
    ancestors = parts[0].split()
    responses = parts[1].split()
    labels = parts[2].split()
    if len(responses) != len(labels):
        raise ValueError(
            f"SARC line {line_number} has {len(responses)} responses and {len(labels)} labels."
        )
    if not responses:
        raise ValueError(f"SARC line {line_number} has no response id.")
    context_turns = [_lookup_turn(comment_id, line_number, comments) for comment_id in ancestors]
    examples: list[CanonicalExample] = []
    for response_id, label_text in zip(responses, labels):
        if label_text not in {"0", "1"}:
            raise ValueError(f"SARC line {line_number} label {label_text!r} is not 0 or 1.")
        response = _require_comment(response_id, line_number, comments)
        text = str(response["text"])
        if not text.strip():
            raise ValueError(f"SARC response {response_id} on line {line_number} has empty text.")
        subreddit = response.get("subreddit", "")
        examples.append(
            CanonicalExample(
                example_id=response_id,
                dataset="sarc",
                split=split,
                text=text,
                context_turns=context_turns,
                text_source="gold",
                label=int(label_text),
                label_meaning="sarcasm",
                extra={
                    "ancestor_ids": ancestors,
                    "subreddit": subreddit,
                    "overlap_text": text,
                },
                overlap_key=response_id,
            )
        )
    return examples


def _lookup_turn(comment_id: str, line_number: int, comments: dict[str, dict[str, Any]]) -> dict[str, str]:
    record = _require_comment(comment_id, line_number, comments)
    return {"speaker": "", "text": str(record["text"])}


def _require_comment(comment_id: str, line_number: int, comments: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if comment_id not in comments:
        raise KeyError(f"SARC line {line_number} references missing comment id {comment_id}.")
    return comments[comment_id]
