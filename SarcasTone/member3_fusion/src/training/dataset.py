"""Multimodal dataset for member3_fusion.

Loads MUStARD (and optionally MUStARD++) from:
  - datasets/MUSTARD/sarcasm_data.json          (labels, text, context, speaker)
  - member2_audio/outputs/handoff/mustard_audio_handoff.csv   (acoustic features)
  - member2_audio/outputs/handoff/mustard_audio_quality.csv   (audio quality)
  - member2_audio/outputs/handoff/mustard_audio_predictions.csv (audio preds)
  - (optional) wav2vec2_embeddings.npz / mustard_frame_embeddings.npz

Design principles
-----------------
* Every row has a sarcasm label that came from the actual dataset file.
* Missing modalities are represented by zeros and a boolean mask.
* Acoustic feature columns are aligned positionally to shared.constants.PROSODY_FEATURES
  where available; full 116-column vectors are also stored for the classifier.
* Speaker-aware splitting uses shared.splits utilities.
* The class never fabricates a probability or an embedding.
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# shared utilities (repo-level)
import sys
_REPO = Path(__file__).resolve().parents[5]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from shared.constants import (
    PROSODY_FEATURES,
    PROSODY_DIM,
    QUALITY_FEATURES,
    QUALITY_DIM,
    TEXT_HIDDEN,
    AUDIO_HIDDEN,
)
from shared.splits import assign_speaker_splits, deduplicate_mustard_family
from shared.examples import UtteranceExample

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mapping from MUSTARD++ columns to shared quality/prosody names
# ---------------------------------------------------------------------------
# The handoff CSV has column names from member2 feature extraction.
# We map them to shared.constants.PROSODY_FEATURES where possible.
_HANDOFF_PROSODY_MAP: dict[str, str] = {
    # (handoff col) -> (shared name)
    "pitch_f0_mean":          "f0_mean",
    "pitch_f0_std":           "f0_std",
    "rms_energy_mean":        "rms_mean",
    "rms_energy_std":         "rms_std",
    "spectral_centroid_mean": "spectral_centroid_mean",
    "spectral_centroid_std":  "spectral_centroid_std",
    "spectral_rolloff_mean":  "spectral_rolloff_mean",
    "spectral_rolloff_std":   "spectral_rolloff_std",
    "zero_crossing_rate_mean":"zcr_mean",
    "zero_crossing_rate_std": "zcr_std",
    "duration_seconds":       "duration_sec",
}

_HANDOFF_QUALITY_MAP: dict[str, str] = {
    "clipping_ratio":  "clipping_ratio",
    "silence_ratio":   "silence_ratio",
    "rms_energy":      "rms",
}


# ---------------------------------------------------------------------------
# Data record
# ---------------------------------------------------------------------------

class FusionRecord:
    """One training/evaluation sample with all modalities."""

    __slots__ = (
        "example_id",
        "dataset",
        "text",
        "context",
        "speaker",
        "label",
        "split",
        # Acoustic
        "audio_embedding",        # np.ndarray [768] or None
        "audio_available",        # bool
        "prosody_vector",         # np.ndarray [PROSODY_DIM]
        "acoustic_features_full", # np.ndarray [116] (all handoff cols)
        "quality_vector",         # np.ndarray [QUALITY_DIM]
        "quality_ok",             # bool
        "audio_duration_sec",     # float | None
    )

    def __init__(
        self,
        example_id: str,
        dataset: str,
        text: str,
        context: list[str],
        speaker: str,
        label: int,
        split: str = "train",
        audio_embedding: np.ndarray | None = None,
        audio_available: bool = False,
        prosody_vector: np.ndarray | None = None,
        acoustic_features_full: np.ndarray | None = None,
        quality_vector: np.ndarray | None = None,
        quality_ok: bool = False,
        audio_duration_sec: float | None = None,
    ) -> None:
        self.example_id = example_id
        self.dataset = dataset
        self.text = text
        self.context = context
        self.speaker = speaker
        self.label = int(label)
        self.split = split
        self.audio_embedding = audio_embedding
        self.audio_available = audio_available
        self.prosody_vector = (
            prosody_vector if prosody_vector is not None
            else np.zeros(PROSODY_DIM, dtype=np.float32)
        )
        self.acoustic_features_full = (
            acoustic_features_full if acoustic_features_full is not None
            else np.zeros(116, dtype=np.float32)
        )
        self.quality_vector = (
            quality_vector if quality_vector is not None
            else np.zeros(QUALITY_DIM, dtype=np.float32)
        )
        self.quality_ok = quality_ok
        self.audio_duration_sec = audio_duration_sec


# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------

def _load_mustard_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _load_handoff_csvs(
    handoff_csv: Path,
    quality_csv: Path,
    predictions_csv: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all three member2 handoff CSVs. Index by utterance_id."""
    df_h = pd.read_csv(handoff_csv, dtype={"utterance_id": str})
    df_q = pd.read_csv(quality_csv, dtype={"utterance_id": str})
    df_p = pd.read_csv(predictions_csv, dtype={"utterance_id": str})
    df_h = df_h.set_index("utterance_id")
    df_q = df_q.set_index("utterance_id")
    df_p = df_p.set_index("utterance_id")
    return df_h, df_q, df_p


def _load_wav2vec2_npz(npz_path: Path | None) -> dict[str, np.ndarray] | None:
    """Load pooled Wav2Vec2 embeddings from npz if available."""
    if npz_path is None or not Path(npz_path).is_file():
        return None
    archive = np.load(npz_path, allow_pickle=False)
    ids = archive["utterance_ids"].astype(str)
    embs = archive["embeddings"].astype(np.float32)
    assert embs.shape == (len(ids), 768), (
        f"Expected wav2vec2 embeddings shape (N, 768), got {embs.shape}"
    )
    return dict(zip(ids, embs))


def _build_prosody_vector(row: pd.Series) -> np.ndarray:
    """Build a PROSODY_DIM vector aligned to shared.constants.PROSODY_FEATURES."""
    # Build reverse map: shared_name -> handoff col
    rev = {v: k for k, v in _HANDOFF_PROSODY_MAP.items()}
    vec = np.zeros(PROSODY_DIM, dtype=np.float32)
    for i, feat_name in enumerate(PROSODY_FEATURES):
        handoff_col = rev.get(feat_name)
        if handoff_col and handoff_col in row.index:
            val = row[handoff_col]
            if np.isfinite(float(val)):
                vec[i] = float(val)
    return vec


def _build_quality_vector(qrow: pd.Series) -> np.ndarray:
    """Build a QUALITY_DIM vector aligned to shared.constants.QUALITY_FEATURES."""
    # shared: snr_db_estimate, clipping_ratio, silence_ratio,
    #         duration_in_range, rms, spectral_flatness
    vec = np.zeros(QUALITY_DIM, dtype=np.float32)
    # Map what we have
    mapping = {
        "clipping_ratio": 1,
        "silence_ratio": 2,
        "rms_energy": 4,
    }
    for col, idx in mapping.items():
        if col in qrow.index:
            val = qrow[col]
            if pd.notna(val) and np.isfinite(float(val)):
                vec[idx] = float(val)
    # duration_in_range (index 3): 1.0 if audio_valid is True
    if "audio_valid" in qrow.index:
        vec[3] = 1.0 if bool(qrow["audio_valid"]) else 0.0
    return vec


def _build_acoustic_full(row: pd.Series, feature_cols: list[str]) -> np.ndarray:
    """Build 116-dim acoustic feature vector from handoff CSV row."""
    vec = np.zeros(len(feature_cols), dtype=np.float32)
    for i, col in enumerate(feature_cols):
        val = row.get(col, 0.0)
        try:
            fval = float(val)
            vec[i] = fval if np.isfinite(fval) else 0.0
        except (TypeError, ValueError):
            vec[i] = 0.0
    return vec


# ---------------------------------------------------------------------------
# Main loader function
# ---------------------------------------------------------------------------

ACOUSTIC_FEATURE_COLS: list[str] = []  # populated on first load


def load_mustard_fusion_records(
    mustard_json_path: Path | str,
    handoff_csv_path: Path | str,
    quality_csv_path: Path | str,
    predictions_csv_path: Path | str,
    wav2vec2_npz_path: Path | str | None = None,
    *,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    split_strategy: str = "speaker",
) -> tuple[dict[str, list[FusionRecord]], int]:
    """Build train/val/test FusionRecord dicts from MUStARD + member2 handoff.

    Returns:
        (splits_dict, acoustic_dim) where splits_dict has keys "train", "val", "test"
        and acoustic_dim is the number of acoustic feature columns detected.
    """
    global ACOUSTIC_FEATURE_COLS

    mustard_json_path = Path(mustard_json_path)
    handoff_csv_path = Path(handoff_csv_path)
    quality_csv_path = Path(quality_csv_path)
    predictions_csv_path = Path(predictions_csv_path)

    log.info("Loading MUStARD JSON from %s", mustard_json_path)
    mustard_data = _load_mustard_json(mustard_json_path)
    log.info("  -> %d utterances", len(mustard_data))

    log.info("Loading member2 handoff CSVs")
    df_h, df_q, df_p = _load_handoff_csvs(
        handoff_csv_path, quality_csv_path, predictions_csv_path
    )

    # Identify acoustic feature columns (all numeric columns except meta)
    _meta_cols = {
        "utterance_id", "speaker", "sarcasm",
        "audio_sarcasm_prediction", "audio_sarcasm_label",
        "audio_sarcasm_decision_score", "sarcasm_score_type",
        "predicted_emotion", "emotion_decision_scores", "emotion_score_type",
        "pooled_embedding_dim", "frame_count", "frame_embedding_dim",
        "audio_duration_seconds", "sample_rate", "channels",
        "peak_amplitude", "rms_energy", "clipping_ratio", "silence_ratio",
        "audio_valid", "audio_warnings", "quality_status",
    }
    ACOUSTIC_FEATURE_COLS = [
        c for c in df_h.columns
        if c not in _meta_cols and df_h[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]
    ]
    log.info("  -> %d acoustic feature columns", len(ACOUSTIC_FEATURE_COLS))

    wav2vec2_map = _load_wav2vec2_npz(wav2vec2_npz_path)
    if wav2vec2_map:
        log.info("Loaded wav2vec2 embeddings for %d utterances", len(wav2vec2_map))
    else:
        log.info("No wav2vec2 npz file; using zero audio embeddings (feature-only mode)")

    # Build UtteranceExample list for speaker-aware splitting
    ue_list: list[UtteranceExample] = []
    for uid, record in mustard_data.items():
        ue = UtteranceExample(
            example_id=str(uid),
            dataset="mustard",
            text=record["utterance"],
            speaker=record.get("speaker", ""),
            task="sarcasm",
            context=record.get("context", []),
            context_speakers=record.get("context_speakers", []),
            label=1 if record["sarcasm"] else 0,
            show=record.get("show"),
        )
        ue_list.append(ue)

    # Split
    if split_strategy == "speaker":
        buckets = assign_speaker_splits(
            ue_list, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
        )
    else:
        # Stratified fallback using sklearn
        from sklearn.model_selection import train_test_split
        labels = [ue.label for ue in ue_list]
        train_val, test_ue = train_test_split(
            ue_list, test_size=test_ratio, stratify=labels, random_state=seed
        )
        tv_labels = [ue.label for ue in train_val]
        relative_val = val_ratio / (1.0 - test_ratio)
        train_ue, val_ue = train_test_split(
            train_val, test_size=relative_val, stratify=tv_labels, random_state=seed
        )
        buckets = {"train": train_ue, "val": val_ue, "test": test_ue}

    # Build FusionRecords
    result: dict[str, list[FusionRecord]] = {"train": [], "val": [], "test": []}

    missing_in_handoff = 0
    for split_name, examples in buckets.items():
        for ue in examples:
            uid = ue.example_id

            # --- Audio embedding ---
            if wav2vec2_map and uid in wav2vec2_map:
                audio_emb = wav2vec2_map[uid]
                audio_available = True
            else:
                audio_emb = None
                audio_available = False

            # --- Acoustic features from handoff CSV ---
            if uid in df_h.index:
                hrow = df_h.loc[uid]
                prosody_vec = _build_prosody_vector(hrow)
                acoustic_full = _build_acoustic_full(hrow, ACOUSTIC_FEATURE_COLS)
                # Use wav2vec2 pooled cols if no npz available
                if not audio_available:
                    # We have no pooled embedding; mark unavailable
                    audio_emb = None
                    audio_available = False
            else:
                prosody_vec = None
                acoustic_full = None
                missing_in_handoff += 1

            # --- Audio quality ---
            if uid in df_q.index:
                qrow = df_q.loc[uid]
                quality_vec = _build_quality_vector(qrow)
                quality_ok = bool(qrow.get("audio_valid", False))
                audio_dur = float(qrow.get("audio_duration_seconds", 0.0) or 0.0)
            else:
                quality_vec = None
                quality_ok = False
                audio_dur = None

            rec = FusionRecord(
                example_id=uid,
                dataset="mustard",
                text=ue.text,
                context=ue.context,
                speaker=ue.speaker,
                label=ue.label,
                split=split_name,
                audio_embedding=audio_emb,
                audio_available=audio_available,
                prosody_vector=prosody_vec,
                acoustic_features_full=acoustic_full,
                quality_vector=quality_vec,
                quality_ok=quality_ok,
                audio_duration_sec=audio_dur,
            )
            result[split_name].append(rec)

    if missing_in_handoff > 0:
        log.warning(
            "%d utterances had no entry in member2 handoff CSV", missing_in_handoff
        )

    for split_name, recs in result.items():
        n_sarc = sum(r.label for r in recs)
        log.info(
            "Split '%s': %d examples (%d sarcastic, %d non-sarcastic)",
            split_name, len(recs), n_sarc, len(recs) - n_sarc,
        )

    return result


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class FusionDataset(Dataset):
    """PyTorch dataset wrapping FusionRecord lists.

    Text is returned as raw string; the text encoder interface handles
    tokenisation. Audio embedding and prosody/quality vectors are pre-loaded.

    Tensors returned
    ----------------
    audio_embedding:   float32 [AUDIO_HIDDEN]   (zeros if unavailable)
    audio_available:   bool scalar
    prosody:           float32 [PROSODY_DIM]
    acoustic_full:     float32 [acoustic_dim]
    quality:           float32 [QUALITY_DIM]
    quality_ok:        bool scalar
    label:             int64 scalar
    """

    def __init__(
        self,
        records: list[FusionRecord],
        acoustic_dim: int | None = None,
    ) -> None:
        self.records = records
        self.acoustic_dim = acoustic_dim or (
            len(ACOUSTIC_FEATURE_COLS) if ACOUSTIC_FEATURE_COLS else 116
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]

        # Audio embedding
        if rec.audio_embedding is not None:
            audio_emb = torch.tensor(rec.audio_embedding, dtype=torch.float32)
        else:
            audio_emb = torch.zeros(AUDIO_HIDDEN, dtype=torch.float32)

        prosody = torch.tensor(rec.prosody_vector, dtype=torch.float32)
        quality = torch.tensor(rec.quality_vector, dtype=torch.float32)

        # Acoustic full (pad/truncate to acoustic_dim)
        af = rec.acoustic_features_full
        if af is None:
            af = np.zeros(self.acoustic_dim, dtype=np.float32)
        if len(af) < self.acoustic_dim:
            af = np.pad(af, (0, self.acoustic_dim - len(af)))
        elif len(af) > self.acoustic_dim:
            af = af[: self.acoustic_dim]
        acoustic_full = torch.tensor(af, dtype=torch.float32)

        return {
            "example_id": rec.example_id,
            "text": rec.text,
            "context": rec.context,                        # list[str]
            "speaker": rec.speaker,
            "audio_embedding": audio_emb,                  # [AUDIO_HIDDEN]
            "audio_available": torch.tensor(rec.audio_available, dtype=torch.bool),
            "prosody": prosody,                            # [PROSODY_DIM]
            "acoustic_full": acoustic_full,                # [acoustic_dim]
            "quality": quality,                            # [QUALITY_DIM]
            "quality_ok": torch.tensor(rec.quality_ok, dtype=torch.bool),
            "label": torch.tensor(rec.label, dtype=torch.long),
        }


def collate_fn(batch: list[dict]) -> dict:
    """Custom collate that handles variable-length context lists."""
    keys_tensor = [
        "audio_embedding", "audio_available",
        "prosody", "acoustic_full",
        "quality", "quality_ok", "label",
    ]
    out: dict = {}
    for key in keys_tensor:
        out[key] = torch.stack([b[key] for b in batch])

    out["example_ids"] = [b["example_id"] for b in batch]
    out["texts"] = [b["text"] for b in batch]
    out["contexts"] = [b["context"] for b in batch]   # list[list[str]]
    out["speakers"] = [b["speaker"] for b in batch]
    return out
