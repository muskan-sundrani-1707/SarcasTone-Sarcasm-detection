
import os

os.environ["HF_HOME"] = r"E:\SarcasTone_Cache"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from transformers import (
    Wav2Vec2Model,
    Wav2Vec2FeatureExtractor
)

from SarcasTone.member2_audio.src.feature_extraction.acoustic_features import (
    extract_features
)

from SarcasTone.member2_audio.src.audio_intelligence.audio_preprocessing import (
    prepare_audio_for_wav2vec2
)

from SarcasTone.member2_audio.src.audio_intelligence.audio_quality import (
    get_audio_quality as _get_audio_quality
)


ROOT = Path(__file__).resolve().parents[2]

EMOTION_MODEL_PATH = (
    ROOT / "checkpoints"
    / "ravdess_emotion_svm_rbf.joblib"
)

SARCASM_MODEL_PATH = (
    ROOT / "checkpoints" / "sarcasm"
    / "wav2vec2_sarcasm_svm.joblib"
)

MODEL_NAME = "facebook/wav2vec2-base"

torch.set_num_threads(2)


# ==================================================
# LAZY MODEL STORAGE
# ==================================================

_emotion_bundle = None
_sarcasm_model = None
_wav2vec_model = None
_wav2vec_processor = None


def load_emotion_model():

    global _emotion_bundle

    if _emotion_bundle is None:
        _emotion_bundle = joblib.load(
            EMOTION_MODEL_PATH
        )

    return _emotion_bundle


def load_sarcasm_model():

    global _sarcasm_model

    if _sarcasm_model is None:
        _sarcasm_model = joblib.load(
            SARCASM_MODEL_PATH
        )

    return _sarcasm_model


def load_wav2vec2():

    global _wav2vec_model, _wav2vec_processor

    if _wav2vec_model is None:

        _wav2vec_processor = (
            Wav2Vec2FeatureExtractor.from_pretrained(
                MODEL_NAME
            )
        )

        _wav2vec_model = Wav2Vec2Model.from_pretrained(
            MODEL_NAME,
            low_cpu_mem_usage=True
        )

        _wav2vec_model.eval()
        _wav2vec_model.to("cpu")

    return _wav2vec_model, _wav2vec_processor


# ==================================================
# HANDCRAFTED ACOUSTIC FEATURES
# ==================================================

def extract_audio_features(audio_path):

    raw_features = extract_features(
        Path(audio_path)
    )

    emotion_bundle = load_emotion_model()

    feature_columns = emotion_bundle["feature_columns"]

    feature_values = {
        column: raw_features[column]
        for column in feature_columns
    }

    return feature_values


# ==================================================
# WAV2VEC2 EMBEDDING
# ==================================================

def get_audio_embedding(audio_path):

    waveform, sr, metadata = (
        prepare_audio_for_wav2vec2(audio_path)
    )

    model, processor = load_wav2vec2()

    inputs = processor(
        waveform,
        sampling_rate=sr,
        return_tensors="pt"
    )

    with torch.inference_mode():

        output = model(
            inputs.input_values
        )

        embedding = (
            output.last_hidden_state
            .mean(dim=1)
            .squeeze(0)
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    return embedding


# ==================================================
# FRAME-LEVEL WAV2VEC2 SEQUENCE EMBEDDING
# ==================================================

def get_audio_sequence_embedding(audio_path):

    """
    Extract frame-level Wav2Vec2 embeddings.

    Args:
        audio_path: Path to an audio file.

    Returns:
        np.ndarray:
            Shape (num_frames, 768)
            dtype float32

    Each row represents one temporal feature frame.
    """

    waveform, sr, metadata = (
        prepare_audio_for_wav2vec2(audio_path)
    )

    model, processor = load_wav2vec2()

    inputs = processor(
        waveform,
        sampling_rate=sr,
        return_tensors="pt"
    )

    with torch.inference_mode():

        output = model(
            inputs.input_values
        )

        sequence_embedding = (
            output.last_hidden_state
            .squeeze(0)
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    if sequence_embedding.ndim != 2:
        raise ValueError(
            "Expected a 2D frame-level embedding."
        )

    if sequence_embedding.shape[1] != 768:
        raise ValueError(
            f"Expected 768 dimensions, got "
            f"{sequence_embedding.shape[1]}"
        )

    if not np.isfinite(sequence_embedding).all():
        raise ValueError(
            "Embedding contains NaN or infinite values."
        )

    return sequence_embedding


# ==================================================
# AUDIO SARCASM PREDICTION
# ==================================================

def predict_audio(audio_path):

    embedding = get_audio_embedding(audio_path)

    model = load_sarcasm_model()

    input_embedding = embedding.reshape(1, -1)

    prediction = int(
        model.predict(input_embedding)[0]
    )

    decision_score = float(
        model.decision_function(input_embedding)[0]
    )

    return {
        "prediction": prediction,

        "label": (
            "sarcastic"
            if prediction == 1
            else "non-sarcastic"
        ),

        "decision_score": decision_score,

        "score_note": (
            "Raw SVM decision score, "
            "not a calibrated probability."
        )
    }


# ==================================================
# EMOTION PREDICTION
# ==================================================

def predict_emotion(audio_path):

    raw_features = extract_audio_features(
        audio_path
    )

    bundle = load_emotion_model()

    feature_columns = bundle["feature_columns"]

    feature_frame = pd.DataFrame(
        [[raw_features[c] for c in feature_columns]],
        columns=feature_columns
    )

    model = bundle["model"]

    prediction = model.predict(
        feature_frame
    )[0]

    result = {
        "emotion": str(prediction)
    }

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(
            feature_frame
        )[0]

        result["confidence"] = float(
            np.max(probabilities)
        )

        result["probabilities"] = {
            str(label): float(probability)
            for label, probability in zip(
                model.classes_,
                probabilities
            )
        }

        result["score_type"] = (
            "Model probability estimates"
        )

    else:

        decision_scores = model.decision_function(
            feature_frame
        )

        result["decision_scores"] = (
            np.asarray(decision_scores).tolist()
        )

        result["score_type"] = (
            "Raw SVM decision scores, "
            "not probabilities or confidence."
        )

    return result


# ==================================================
# AUDIO QUALITY
# ==================================================

def get_audio_quality(audio_path):

    return _get_audio_quality(audio_path)


# Backward-compatible alias

def assess_audio_quality(audio_path):

    return get_audio_quality(audio_path)
