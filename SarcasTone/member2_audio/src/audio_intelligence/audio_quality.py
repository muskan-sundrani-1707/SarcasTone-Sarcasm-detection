
import numpy as np

from SarcasTone.member2_audio.src.audio_processing.audio_loader import load_audio


def get_audio_quality(audio_path):
    """
    Assess basic audio quality indicators.
    """

    waveform, sr, metadata = load_audio(
        audio_path,
        mono=True,
        normalize=False
    )

    waveform = np.asarray(waveform, dtype=np.float32)

    peak = float(np.max(np.abs(waveform)))
    rms = float(np.sqrt(np.mean(waveform ** 2)))

    clipping_ratio = float(
        np.mean(np.abs(waveform) >= 0.999)
    )

    silence_ratio = float(
        np.mean(np.abs(waveform) < 0.005)
    )

    snr_db = None

    # Approximate signal-to-noise indicator.
    # This is not a true noise-floor SNR estimate.
    if rms > 0:
        snr_db = float(
            20 * np.log10(rms / max(peak * 0.01, 1e-10))
        )

    duration = metadata["duration_seconds"]

    warnings = []

    if duration < 0.5:
        warnings.append("Audio is very short.")

    if clipping_ratio > 0.01:
        warnings.append("Possible audio clipping detected.")

    if silence_ratio > 0.5:
        warnings.append("High proportion of near-silent samples.")

    if rms < 0.005:
        warnings.append("Very low audio energy.")

    return {
        "duration_seconds": float(duration),
        "sample_rate": int(sr),
        "channels": int(metadata["original_channels"]),
        "peak_amplitude": peak,
        "rms_energy": rms,
        "clipping_ratio": clipping_ratio,
        "silence_ratio": silence_ratio,
        "approximate_level_indicator_db": snr_db,
        "warnings": warnings,
        "is_valid": len(warnings) == 0
    }
