
from pathlib import Path
import argparse
import csv
import numpy as np
import librosa

from src.audio_processing.audio_loader import load_audio

ROOT = Path(__file__).resolve().parents[2]

RAVDESS_DIR = (
    ROOT / "data" / "raw" / "RAVDESS"
    / "Audio_Speech_Actors_01-24"
)

OUTPUT_FILE = (
    ROOT / "data" / "processed"
    / "ravdess_acoustic_features.csv"
)

N_MFCC = 13
N_MELS = 40
N_FFT = 2048
HOP_LENGTH = 512


def mean_std(values, prefix):
    values = np.asarray(values, dtype=np.float64)

    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values))
    }


def extract_features(audio_path):

    waveform, sr, metadata = load_audio(
        audio_path,
        mono=True,
        normalize=False
    )

    if len(waveform) < N_FFT:
        raise ValueError("Audio is too short for feature extraction.")

    features = {
        "sample_id": audio_path.stem,
        "sample_rate": sr,
        "duration_seconds": metadata["duration_seconds"]
    }

    # 1. MFCC features
    mfcc = librosa.feature.mfcc(
        y=waveform,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    for i in range(N_MFCC):
        features.update(
            mean_std(mfcc[i], f"mfcc_{i+1}")
        )

    # 2. Mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    for i in range(N_MELS):
        features.update(
            mean_std(mel_db[i], f"mel_{i+1}")
        )

    # 3. Spectral features
    centroid = librosa.feature.spectral_centroid(
        y=waveform,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )[0]

    rolloff = librosa.feature.spectral_rolloff(
        y=waveform,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )[0]

    zcr = librosa.feature.zero_crossing_rate(
        waveform,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH
    )[0]

    rms = librosa.feature.rms(
        y=waveform,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH
    )[0]

    features.update(mean_std(centroid, "spectral_centroid"))
    features.update(mean_std(rolloff, "spectral_rolloff"))
    features.update(mean_std(zcr, "zero_crossing_rate"))
    features.update(mean_std(rms, "rms_energy"))

    # 4. Pitch estimation using YIN
    pitch = librosa.yin(
        waveform,
        fmin=65,
        fmax=500,
        sr=sr,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH
    )

    features.update(mean_std(pitch, "pitch_f0"))

    return features


def run_extraction(limit=None):

    audio_files = sorted(
        RAVDESS_DIR.rglob("*.wav")
    )

    if not audio_files:
        raise FileNotFoundError(
            f"No audio files found: {RAVDESS_DIR}"
        )

    if limit is not None:
        audio_files = audio_files[:limit]

    records = []
    failures = []

    print(f"Processing {len(audio_files)} audio files...")

    for index, audio_path in enumerate(audio_files, start=1):

        try:
            record = extract_features(audio_path)
            records.append(record)

        except Exception as error:
            failures.append({
                "file": audio_path.name,
                "error": str(error)
            })

        if index % 50 == 0 or index == len(audio_files):
            print(
                f"Processed {index}/{len(audio_files)}"
            )

    if not records:
        raise RuntimeError("No audio features were extracted.")

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = list(records[0].keys())

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(records)

    print("\nACOUSTIC FEATURE EXTRACTION")
    print("=" * 45)
    print("Requested files:", len(audio_files))
    print("Successful:", len(records))
    print("Failed:", len(failures))
    print("Feature dimensions:", len(fieldnames) - 1)

    print("\nOutput saved:")
    print(OUTPUT_FILE)

    if failures:
        print("\nFailed recordings:")
        for failure in failures[:10]:
            print(failure)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of files for testing."
    )

    args = parser.parse_args()

    run_extraction(limit=args.limit)
