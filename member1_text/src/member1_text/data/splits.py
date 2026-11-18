"""Reproducible id splits. Official SARC and SemEval files are not reshuffled."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold

from member1_text.canonicalize import CanonicalExample, normalize_overlap_text


def _labels(examples: Sequence[CanonicalExample]) -> list[int]:
    labels: list[int] = []
    for example in examples:
        label = example.label
        if isinstance(label, bool):
            labels.append(int(label))
        elif isinstance(label, float) and not float(label).is_integer():
            raise ValueError("Stratified folds need discrete labels.")
        else:
            labels.append(int(label))
    return labels


def stratified_kfold(
    examples: Sequence[CanonicalExample],
    *,
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[list[str], list[str]]]:
    """Stratified folds. Each id appears in exactly one validation fold."""
    if len(examples) < n_splits:
        raise ValueError(f"Need at least {n_splits} examples for {n_splits} folds.")
    ids = [example.example_id for example in examples]
    if len(ids) != len(set(ids)):
        raise ValueError("example_id values must be unique before folding.")
    labels = np.asarray(_labels(examples))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds: list[tuple[list[str], list[str]]] = []
    positions = np.arange(len(examples))
    for train_index, valid_index in splitter.split(positions, labels):
        train_ids = [ids[index] for index in train_index]
        valid_ids = [ids[index] for index in valid_index]
        folds.append((train_ids, valid_ids))
    return folds


def speaker_grouped_split(
    examples: Sequence[CanonicalExample],
    *,
    seed: int = 42,
    test_size: float = 0.2,
    speaker_key: str = "speaker",
) -> tuple[list[str], list[str]]:
    """Hold out speakers so no speaker appears on both sides."""
    ids = [example.example_id for example in examples]
    groups = [str(example.extra.get(speaker_key, "")) for example in examples]
    if len(set(groups)) < 2:
        raise ValueError("Speaker-grouped split needs at least two speakers.")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_index, holdout_index = next(splitter.split(np.arange(len(examples)), groups=groups))
    return [ids[index] for index in train_index], [ids[index] for index in holdout_index]


def stratified_holdout(
    examples: Sequence[CanonicalExample],
    *,
    seed: int = 42,
    valid_size: float = 0.1,
) -> tuple[list[str], list[str]]:
    """One stratified slice. Used for early stopping inside an official train file."""
    if not 0.0 < valid_size < 1.0:
        raise ValueError("valid_size must be between 0 and 1.")
    n_splits = max(2, int(round(1.0 / valid_size)))
    n_splits = min(n_splits, len(examples))
    labels = _labels(examples)
    smallest = min(labels.count(label) for label in set(labels))
    while n_splits > smallest and n_splits > 2:
        n_splits -= 1
    folds = stratified_kfold(examples, n_splits=n_splits, seed=seed)
    valid_ids = folds[0][1]
    valid_set = set(valid_ids)
    train_ids = [example.example_id for example in examples if example.example_id not in valid_set]
    return train_ids, valid_ids


def subset_by_ids(
    examples: Sequence[CanonicalExample],
    ids: Sequence[str],
    *,
    split_name: str | None = None,
) -> list[CanonicalExample]:
    by_id = {example.example_id: example for example in examples}
    missing = [example_id for example_id in ids if example_id not in by_id]
    if missing:
        raise KeyError(f"Unknown example ids: {missing[:5]}")
    selected: list[CanonicalExample] = []
    for example_id in ids:
        example = by_id[example_id]
        if split_name is not None and example.split != split_name:
            example = CanonicalExample(
                example_id=example.example_id,
                dataset=example.dataset,
                split=split_name,
                text=example.text,
                context_turns=list(example.context_turns),
                text_source=example.text_source,
                label=example.label,
                label_meaning=example.label_meaning,
                extra=dict(example.extra),
                overlap_key=example.overlap_key,
            )
        selected.append(example)
    return selected


def drop_mustard_overlap(
    mustard: Sequence[CanonicalExample],
    auxiliary: Sequence[CanonicalExample],
) -> tuple[list[CanonicalExample], int]:
    """Remove auxiliary rows whose scene id or normalized text also occurs in MUStARD."""
    mustard_ids = {example.example_id for example in mustard}
    mustard_keys = {example.overlap_key for example in mustard if example.overlap_key}
    mustard_text = {
        normalize_overlap_text(str(example.extra.get("overlap_text", example.text)))
        for example in mustard
    }
    kept: list[CanonicalExample] = []
    removed = 0
    for example in auxiliary:
        scene = str(example.extra.get("scene", example.overlap_key))
        text = normalize_overlap_text(str(example.extra.get("overlap_text", example.text)))
        if (
            example.example_id in mustard_ids
            or example.overlap_key in mustard_ids
            or scene in mustard_ids
            or example.overlap_key in mustard_keys
            or text in mustard_text
        ):
            removed += 1
            continue
        kept.append(example)
    return kept, removed


def median_epoch(epochs: Sequence[int]) -> int:
    """Middle value of the sorted per-fold best epochs. Five folds use index 2."""
    if not epochs:
        raise ValueError("Cannot take the median of an empty epoch list.")
    ordered = sorted(int(epoch) for epoch in epochs)
    return ordered[len(ordered) // 2]
