"""Tests for multimodal synchronization between Member 1 (Text) and Member 2 (Audio)."""

from __future__ import annotations

import unittest
from pathlib import Path
import numpy as np

from shared.constants import (
    AUDIO_HIDDEN,
    PROSODY_DIM,
    PROSODY_FEATURES,
    QUALITY_DIM,
    QUALITY_FEATURES,
    TARGET_SAMPLE_RATE,
    TEXT_HIDDEN,
)
from shared.sync import sync_utterance, verify_corpus_sync


class MultimodalSyncTest(unittest.TestCase):

    def test_corpus_synchronization_690_utterances(self) -> None:
        """Verify that Member 1 text data and Member 2 audio manifest are 100% aligned."""
        report = verify_corpus_sync()
        self.assertTrue(report["is_synced"], msg=f"Sync report: {report}")
        self.assertEqual(report["aligned_records"], 690)
        self.assertEqual(report["total_member1_records"], 690)
        self.assertEqual(report["total_member2_records"], 690)
        self.assertEqual(len(report["missing_in_member1"]), 0)
        self.assertEqual(len(report["missing_in_member2"]), 0)
        self.assertEqual(len(report["label_mismatches"]), 0)
        self.assertEqual(len(report["speaker_mismatches"]), 0)

    def test_contract_feature_dimensions(self) -> None:
        """Verify dimensional contract compliance for prosody, quality, and embeddings."""
        self.assertEqual(len(PROSODY_FEATURES), 32)
        self.assertEqual(PROSODY_DIM, 32)
        self.assertEqual(len(QUALITY_FEATURES), 6)
        self.assertEqual(QUALITY_DIM, 6)
        self.assertEqual(TEXT_HIDDEN, 768)
        self.assertEqual(AUDIO_HIDDEN, 768)
        self.assertEqual(TARGET_SAMPLE_RATE, 16000)

    def test_sync_utterance_multimodal(self) -> None:
        """Verify synchronized feature extraction for a paired text and audio utterance."""
        fixture_wav = Path(__file__).resolve().parents[1] / "member2_audio" / "tests" / "fixtures" / "test_sample.wav"
        if not fixture_wav.is_file():
            # Create synthetic test WAV if not present
            import wave
            import struct
            import math

            fixture_wav.parent.mkdir(parents=True, exist_ok=True)
            sr = 16000
            dur = 1.0
            frames = int(sr * dur)
            with wave.open(str(fixture_wav), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                packed = bytearray()
                for i in range(frames):
                    val = int(10000 * math.sin(2 * math.pi * 440 * i / sr))
                    packed.extend(struct.pack("<h", val))
                wf.writeframes(bytes(packed))

        bundle = sync_utterance(
            text="Oh, absolutely brilliant observation!",
            audio_path=fixture_wav,
            context=["Do you really think it will work?"],
            speaker="CHANDLER",
            example_id="test_sync_01",
        )

        self.assertEqual(bundle.example_id, "test_sync_01")
        self.assertEqual(bundle.speaker, "CHANDLER")
        self.assertTrue(bundle.has_text)
        self.assertTrue(bundle.has_audio)
        self.assertEqual(bundle.sample_rate_hz, 16000)

        # Check prosody vector & features
        self.assertEqual(len(bundle.prosody), 32)
        for feat in PROSODY_FEATURES:
            self.assertIn(feat, bundle.prosody_features)

        # Check quality vector & features
        self.assertEqual(len(bundle.quality), 6)
        for qfeat in QUALITY_FEATURES:
            self.assertIn(qfeat, bundle.quality_features)

        # Verify quality features are sane
        self.assertAlmostEqual(bundle.quality_features["duration_in_range"], 1.0)
        self.assertGreater(bundle.quality_features["rms"], 0.0)

    def test_sync_utterance_text_only(self) -> None:
        """Verify fallback behavior when only text is provided."""
        bundle = sync_utterance(
            text="Yeah, right.",
            audio_path=None,
            example_id="text_only_01",
        )
        self.assertTrue(bundle.has_text)
        self.assertFalse(bundle.has_audio)
        self.assertEqual(len(bundle.prosody), 32)
        self.assertEqual(len(bundle.quality), 6)
        self.assertTrue(all(v == 0.0 for v in bundle.prosody))
        self.assertTrue(all(v == 0.0 for v in bundle.quality))


if __name__ == "__main__":
    unittest.main()
