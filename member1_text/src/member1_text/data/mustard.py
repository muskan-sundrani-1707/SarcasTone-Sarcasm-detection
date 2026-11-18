"""MUStARD loader.

The JSON object is one clip per key. ``sarcasm`` is the label. The id prefix
is not a label. Context length must match ``context_speakers``.
"""

from __future__ import annotations

import json
import pickle
import urllib.request
from pathlib import Path
from typing import Any

from member1_text.canonicalize import (
    CanonicalExample,
    DatasetNotAvailableError,
    normalize_overlap_text,
    require_local_file,
)
from member1_text.config import PACKAGE_ROOT, load_config, resolve_repo_path
from member1_text.provenance import build_provenance, write_provenance

SOURCE_URL = "https://github.com/soujanyaporia/MUStARD"
TEXT_URL = "https://raw.githubusercontent.com/soujanyaporia/MUStARD/master/data/sarcasm_data.json"
PICKLE_URL = "https://raw.githubusercontent.com/soujanyaporia/MUStARD/master/data/split_indices.p"
VIDEO_URL = "https://huggingface.co/datasets/MichiganNLP/MUStARD"
CITATION = (
    "Santiago Castro, Devamanyu Hazarika, Veronica Perez-Rosas, Roger Zimmermann, "
    "Rada Mihalcea, and Soujanya Poria. Towards Multimodal Sarcasm Detection "
    "(An Obviously Perfect Paper). ACL 2019."
)
LICENSE_NOTE = "MIT license on the GitHub text release."
REQUIRED_KEYS = ("utterance", "speaker", "context", "context_speakers", "show", "sarcasm")


def _dataset_config() -> dict[str, Any]:
    return load_config()["datasets"]["mustard"]


def default_text_path() -> Path:
    return resolve_repo_path(_dataset_config()["local_text_path"])


def load_mustard(path: Path | None = None, *, split: str = "all") -> list[CanonicalExample]:
    """Load MUStARD clips. A missing file raises and does not invent rows."""
    config = _dataset_config()
    source = path if path is not None else default_text_path()
    json_path = require_local_file("MUStARD", source, config["source_url"])
    with json_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(
            "MUStARD sarcasm_data.json must be an object keyed by utterance id. "
            f"Got {type(payload).__name__}."
        )
    examples: list[CanonicalExample] = []
    for example_id, record in payload.items():
        examples.append(_record_to_example(str(example_id), record, split=split))
    if not examples:
        raise ValueError(f"MUStARD file {json_path} contained no records.")
    return examples


def _record_to_example(example_id: str, record: Any, *, split: str) -> CanonicalExample:
    if not isinstance(record, dict):
        raise ValueError(f"MUStARD record {example_id} is not an object.")
    missing = [key for key in REQUIRED_KEYS if key not in record]
    if missing:
        raise ValueError(f"MUStARD record {example_id} is missing keys: {missing}")
    context = record["context"]
    speakers = record["context_speakers"]
    if not isinstance(context, list) or not isinstance(speakers, list):
        raise ValueError(f"MUStARD record {example_id} context fields must be lists.")
    if len(context) != len(speakers):
        raise ValueError(
            f"MUStARD record {example_id} context length {len(context)} "
            f"does not match context_speakers length {len(speakers)}."
        )
    utterance = record["utterance"]
    if not isinstance(utterance, str) or not utterance.strip():
        raise ValueError(f"MUStARD record {example_id} is missing a target utterance.")
    sarcasm = record["sarcasm"]
    if not isinstance(sarcasm, bool):
        raise ValueError(f"MUStARD record {example_id} sarcasm label must be boolean.")
    turns = [
        {"speaker": str(speaker), "text": str(text)}
        for speaker, text in zip(speakers, context)
    ]
    return CanonicalExample(
        example_id=example_id,
        dataset="mustard",
        split=split,
        text=utterance,
        context_turns=turns,
        text_source="gold",
        label=1 if sarcasm else 0,
        label_meaning="sarcasm",
        extra={
            "speaker": str(record["speaker"]),
            "show": str(record["show"]),
            "overlap_text": utterance,
        },
        overlap_key=example_id,
    )


def load_official_split(path: Path | None = None) -> dict[str, Any]:
    """Unpickle the official index file, or report that the split is unavailable."""
    config = _dataset_config()
    pickle_path = path
    if pickle_path is None:
        pickle_path = resolve_repo_path(config["official_split_pickle"])
        if not pickle_path.is_file():
            sibling = default_text_path().with_name("split_indices.p")
            pickle_path = sibling
    if not Path(pickle_path).is_file():
        return {
            "available": False,
            "reason": f"Official split pickle is not present at {pickle_path}.",
            "source_url": config["source_url"],
        }
    loaded = None
    errors: list[str] = []
    for encoding in (None, "latin1"):
        try:
            with Path(pickle_path).open("rb") as handle:
                loaded = pickle.load(handle) if encoding is None else pickle.load(handle, encoding=encoding)
            break
        except Exception as exc:
            errors.append(str(exc))
            loaded = None
    if loaded is None:
        return {
            "available": False,
            "reason": "split_indices.p did not unpickle on this interpreter: " + " | ".join(errors),
            "source_url": config["source_url"],
        }
    interpreted = _interpret_split(loaded)
    interpreted["path"] = str(pickle_path)
    return interpreted


def _interpret_split(loaded: Any) -> dict[str, Any]:
    """Use id lists from the pickle. Unrecognized structures stay unused."""
    if isinstance(loaded, dict):
        folds: dict[str, Any] = {}
        for key, value in loaded.items():
            if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
                folds[str(key)] = list(value)
        if folds:
            return {"available": True, "kind": "id_lists", "folds": folds}
    return {
        "available": False,
        "reason": (
            "split_indices.p unpickled but its structure is not a mapping of utterance id lists. "
            f"Top-level type was {type(loaded).__name__}. The split was not recreated."
        ),
    }


def write_text_provenance(path: Path | None = None) -> Path:
    json_path = require_local_file("MUStARD", path or default_text_path(), SOURCE_URL)
    payload = build_provenance(
        json_path,
        source_url=TEXT_URL,
        license_note=LICENSE_NOTE,
        citation=CITATION,
        extra={"records_file": json_path.name},
    )
    destination = PACKAGE_ROOT / "outputs" / "provenance" / "mustard_sarcasm_data.json"
    return write_provenance(destination, payload)


def download_text(destination: Path | None = None) -> Path:
    """Download the public MIT JSON. This does not download videos."""
    target = destination or default_text_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(TEXT_URL, target)
    except Exception as exc:
        if target.is_file() and target.stat().st_size == 0:
            target.unlink()
        raise DatasetNotAvailableError(
            "MUStARD",
            str(target),
            f"{SOURCE_URL} (download failed: {exc})",
        ) from exc
    write_text_provenance(target)
    return target


def download_audio(destination: Path | None = None) -> Path:
    """Download the utterance video archive. This is the only video downloader."""
    target_dir = destination or (PACKAGE_ROOT / "data" / "mustard" / "audio")
    target_dir.mkdir(parents=True, exist_ok=True)
    archive = target_dir / "mmsd_raw_data.zip"
    url = "https://huggingface.co/datasets/MichiganNLP/MUStARD/resolve/main/mmsd_raw_data.zip"
    try:
        urllib.request.urlretrieve(url, archive)
    except Exception as exc:
        raise DatasetNotAvailableError(
            "MUStARD audio",
            str(archive),
            f"{VIDEO_URL} (download failed: {exc})",
        ) from exc
    return archive
