
from pathlib import Path
import traceback

from SarcasTone.member2_audio.src.audio_intelligence.audio_intelligence import (
    extract_audio_features,
    get_audio_embedding,
    predict_audio,
    predict_emotion,
    get_audio_quality
)

audio_dir = Path("data/processed/mustard/audio")

audio_files = sorted(audio_dir.glob("*.wav"))

if not audio_files:
    raise FileNotFoundError("No MUStARD WAV files found.")

audio_path = audio_files[0]

print("\n" + "=" * 55)
print("SARCAS TONE AUDIO INTEGRATION TEST")
print("=" * 55)

print("\nTest audio:", audio_path.name)

tests = [
    ("Acoustic Features", lambda: extract_audio_features(audio_path)),
    ("Wav2Vec2 Embedding", lambda: get_audio_embedding(audio_path)),
    ("Audio Sarcasm Prediction", lambda: predict_audio(audio_path)),
    ("Emotion Prediction", lambda: predict_emotion(audio_path)),
    ("Audio Quality", lambda: get_audio_quality(audio_path)),
]

for name, function in tests:

    print("\n" + "-" * 45)
    print(name)
    print("-" * 45)

    try:
        result = function()

        if name == "Acoustic Features":
            print("Feature count:", len(result))
            print("Sample:", dict(list(result.items())[:5]))

        elif name == "Wav2Vec2 Embedding":
            print("Shape:", result.shape)
            print("Data type:", result.dtype)

        else:
            print(result)

        print("STATUS: PASSED")

    except Exception as error:
        print("STATUS: FAILED")
        print("Error:", error)
        traceback.print_exc()

print("\n" + "=" * 55)
print("INTEGRATION TEST COMPLETED")
print("=" * 55)
