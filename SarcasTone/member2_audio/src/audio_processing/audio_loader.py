
from pathlib import Path
import numpy as np
import soundfile as sf

from scipy.signal import resample_poly
from math import gcd


def load_audio(
    audio_path,
    mono=True,
    normalize=False,
    target_sr=None
):
    """
    Load an audio file and optionally process it.

    Parameters
    ----------
    audio_path : str or Path
        Path to the audio file.
    mono : bool
        Convert multi-channel audio to mono.
    normalize : bool
        Normalize peak amplitude to 1.0.
    target_sr : int or None
        Optional target sample rate. Original rate is preserved
        when None.

    Returns
    -------
    waveform : numpy.ndarray
        Audio samples as float32.
    sample_rate : int
        Final sample rate.
    metadata : dict
        Audio properties and processing information.
    """

    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    if audio_path.suffix.lower() != ".wav":
        raise ValueError(
            "Currently, only WAV files are supported."
        )

    try:
        waveform, original_sr = sf.read(
            audio_path,
            dtype="float32",
            always_2d=True
        )

    except Exception as error:
        raise ValueError(
            f"Unable to read audio: {error}"
        ) from error

    if waveform.size == 0:
        raise ValueError("Audio file is empty.")

    if not np.isfinite(waveform).all():
        raise ValueError(
            "Audio contains invalid numerical values."
        )

    original_channels = waveform.shape[1]
    original_duration = waveform.shape[0] / original_sr

    # Convert stereo or multi-channel audio to mono
    if mono and original_channels > 1:
        waveform = np.mean(
            waveform, axis=1, keepdims=True
        )

    if mono:
        waveform = waveform[:, 0]

    # Optional amplitude normalization
    peak_before = float(np.max(np.abs(waveform)))

    if normalize and peak_before > 0:
        waveform = waveform / peak_before

    # Optional resampling
    final_sr = original_sr

    if target_sr is not None:

        if not isinstance(target_sr, int) or target_sr <= 0:
            raise ValueError(
                "target_sr must be a positive integer."
            )

        if target_sr != original_sr:

            common = gcd(original_sr, target_sr)

            waveform = resample_poly(
                waveform,
                target_sr // common,
                original_sr // common,
                axis=0
            ).astype(np.float32)

            final_sr = target_sr

    waveform = np.asarray(
        waveform, dtype=np.float32
    )

    final_duration = len(waveform) / final_sr

    metadata = {
        "file_name": audio_path.name,
        "original_sample_rate": int(original_sr),
        "final_sample_rate": int(final_sr),
        "original_channels": int(original_channels),
        "duration_seconds": float(final_duration),
        "original_duration_seconds": float(original_duration),
        "samples": int(len(waveform)),
        "peak_amplitude": float(np.max(np.abs(waveform))),
        "normalized": bool(normalize),
        "mono": bool(mono)
    }

    return waveform, final_sr, metadata
