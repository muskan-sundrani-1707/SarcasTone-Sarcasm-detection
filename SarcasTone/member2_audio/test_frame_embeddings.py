
from pathlib import Path
import numpy as np

from SarcasTone.member2_audio.src.audio_intelligence.audio_intelligence import (
    get_audio_embedding,
    get_audio_sequence_embedding
)

audio_dir = Path("data/processed/mustard/audio")

audio_files = sorted(audio_dir.glob("*.wav"))

if not audio_files:
    raise FileNotFoundError("No audio files found.")

audio_path = audio_files[0]

print("=" * 55)
print("SARCAS TONE FRAME-LEVEL EMBEDDING TEST")
print("=" * 55)

print("Audio:", audio_path.name)

# Extract frame-level sequence
sequence = get_audio_sequence_embedding(audio_path)

print("\nFrame-level embedding:")
print("Shape:", sequence.shape)
print("Data type:", sequence.dtype)
print("Number of frames:", sequence.shape[0])
print("Feature dimensions:", sequence.shape[1])

# Extract existing pooled embedding
pooled = get_audio_embedding(audio_path)

print("\nPooled embedding:")
print("Shape:", pooled.shape)

# Verify consistency
sequence_mean = sequence.mean(axis=0)

print("\nConsistency check:")
print(
    "Mean pooling matches existing embedding:",
    np.allclose(sequence_mean, pooled, atol=1e-5)
)

assert sequence.ndim == 2
assert sequence.shape[1] == 768
assert pooled.shape == (768,)
assert np.isfinite(sequence).all()
assert np.allclose(sequence_mean, pooled, atol=1e-5)

print("\nALL FRAME-LEVEL TESTS PASSED!")
