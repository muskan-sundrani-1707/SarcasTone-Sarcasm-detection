"""Whisper transcription.

The returned text is always marked ``is_ground_truth: false``. Callers must
not write it back into a dataset as a gold transcript.
"""

from __future__ import annotations

from pathlib import Path


def transcribe_audio(audio_path: str | Path, model_size: str = "small") -> dict:
    source = Path(audio_path)
    if not source.is_file():
        raise FileNotFoundError(f"Audio file not found: {source}")
    try:
        return _faster_whisper(source, model_size)
    except ImportError:
        pass
    try:
        return _openai_whisper(source, model_size)
    except ImportError as exc:
        raise ImportError(
            "Install faster-whisper or openai-whisper to transcribe audio. "
            "See requirements-optional.txt."
        ) from exc


def _faster_whisper(source: Path, model_size: str) -> dict:
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(str(source), beam_size=5)
    pieces = []
    log_probs = []
    for segment in segments:
        pieces.append(segment.text)
        if segment.avg_logprob is not None:
            log_probs.append(float(segment.avg_logprob))
    confidence = None
    if log_probs:
        # avg_logprob is negative. Map the mean into (0, 1] for the fusion gate.
        mean_log = sum(log_probs) / len(log_probs)
        confidence = float(pow(2.718281828, mean_log))
        confidence = max(0.0, min(1.0, confidence))
    return {
        "text": " ".join(piece.strip() for piece in pieces).strip(),
        "language": getattr(info, "language", None),
        "duration": getattr(info, "duration", None),
        "asr_confidence": confidence,
        "engine": "faster-whisper",
        "model_size": model_size,
        "is_ground_truth": False,
        "source_path": str(source),
    }


def _openai_whisper(source: Path, model_size: str) -> dict:
    import whisper

    model = whisper.load_model(model_size)
    result = model.transcribe(str(source))
    return {
        "text": (result.get("text") or "").strip(),
        "language": result.get("language"),
        "duration": None,
        "asr_confidence": None,
        "engine": "openai-whisper",
        "model_size": model_size,
        "is_ground_truth": False,
        "source_path": str(source),
    }
