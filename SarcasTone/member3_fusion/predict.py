"""predict.py — CLI and importable API for SarcasTone Fusion inference.

Usage (CLI)
-----------
python member3_fusion/predict.py path/to/audio.wav
python member3_fusion/predict.py path/to/audio.wav --text "It's just great."
python member3_fusion/predict.py path/to/audio.wav --context "What do you think?" --text "Oh, brilliant."
python member3_fusion/predict.py path/to/audio.wav --experiment cross_attn_psid
python member3_fusion/predict.py path/to/audio.wav --temperature 1.3

Usage (Python import)
---------------------
from member3_fusion.predict import predict_multimodal
result = predict_multimodal("audio.wav", text="It's great.", context=["Really?"])
print(result["label"])           # "sarcastic" or "non_sarcastic"
print(result["probabilities"])   # {"non_sarcastic": 0.23, "sarcastic": 0.77}
print(result["uncertain"])       # True/False
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_M3   = _REPO / "member3_fusion"
for p in [str(_REPO), str(_M3), str(_M3 / "src")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.inference.predict_multimodal import predict_multimodal  # noqa: re-export

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("predict")

__all__ = ["predict_multimodal"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SarcasTone multimodal sarcasm prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("audio_path", help="Path to audio file (wav/mp3/mp4)")
    p.add_argument("--text", default=None,
                   help="Ground-truth transcript (skips ASR if provided)")
    p.add_argument("--context", nargs="+", default=None,
                   help="Preceding utterance(s) as space-separated strings")
    p.add_argument("--experiment", default="full_model",
                   help="Experiment name to load checkpoint from (default: full_model)")
    p.add_argument("--checkpoint", default=None,
                   help="Direct path to checkpoint directory (overrides --experiment)")
    p.add_argument("--temperature", type=float, default=None,
                   help="Calibration temperature (default: from model)")
    p.add_argument("--device", default=None, help="torch device (cpu/cuda)")
    p.add_argument("--verbose", action="store_true",
                   help="Print full prediction dict")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve checkpoint
    if args.checkpoint:
        ckpt_dir = Path(args.checkpoint)
    else:
        ckpt_dir = _M3 / "checkpoints" / args.experiment

    result = predict_multimodal(
        audio_path=args.audio_path,
        text=args.text,
        context=args.context,
        checkpoint_dir=ckpt_dir,
        device=args.device,
        temperature=args.temperature,
    )

    if args.verbose:
        print(json.dumps(result, indent=2, default=str))
    else:
        label   = result["label"]
        p_sarc  = result["probabilities"].get("sarcastic", 0.0)
        entropy = result["entropy"]
        flag    = " [UNCERTAIN]" if result["uncertain"] else ""
        print(f"Prediction : {label}{flag}")
        print(f"P(sarcastic): {p_sarc:.4f}")
        print(f"Entropy    : {entropy:.4f} nats")
        if result["text_weight"] is not None:
            print(f"Text gate  : {result['text_weight']:.4f}")
            print(f"Audio gate : {result['audio_weight']:.4f}")
        if result["transcript"] and not result["transcript_is_ground_truth"]:
            print(f"Transcript : {result['transcript']!r}  (ASR)")


if __name__ == "__main__":
    main()
