
import json
import pickle
import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

MUSTARD_DIR = (
    ROOT / "data" / "raw" / "MUSTARD"
    / "MUStARD-master" / "data"
)

FEATURE_FILE = MUSTARD_DIR / "audio_features.p"
ANNOTATION_FILE = MUSTARD_DIR / "sarcasm_data.json"

OUTPUT_FILE = (
    ROOT / "data" / "metadata"
    / "mustard_audio_manifest.csv"
)


def load_mustard():

    with open(FEATURE_FILE, "rb") as f:
        features = pickle.load(f, encoding="latin1")

    with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    if set(features) != set(annotations):
        raise ValueError("Feature and annotation IDs do not match.")

    rows = []

    for sample_id, annotation in annotations.items():

        feature_array = np.asarray(features[sample_id])

        if feature_array.ndim != 2:
            raise ValueError(
                f"Unexpected feature dimensions: {sample_id}"
            )

        rows.append({
            "sample_id": sample_id,
            "speaker": annotation["speaker"],
            "utterance": annotation["utterance"],
            "context": " ".join(annotation["context"]),
            "show": annotation.get("show", ""),
            "sarcasm": int(annotation["sarcasm"]),
            "feature_dimensions": feature_array.shape[0],
            "temporal_bins": feature_array.shape[1],
            "feature_file": str(FEATURE_FILE)
        })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(
            f, fieldnames=rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\nMUStARD LOADING SUCCESSFUL")
    print("-" * 40)
    print("Total samples:", len(rows))
    print("Sarcastic:", sum(r["sarcasm"] for r in rows))
    print("Non-sarcastic:", len(rows) - sum(r["sarcasm"] for r in rows))
    print("Feature dimensions:", rows[0]["feature_dimensions"])
    print("Temporal bins:", rows[0]["temporal_bins"])
    print("Manifest saved:", OUTPUT_FILE)


if __name__ == "__main__":
    load_mustard()
