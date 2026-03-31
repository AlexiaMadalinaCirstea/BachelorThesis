from __future__ import annotations

import json
from pathlib import Path

import joblib


def main() -> None:
    model_path = Path("Datasets/IoT23/processed_test_sample/iot23/rf_baseline/rf_model.joblib")
    out_dir = Path("feature_alignment")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "iot23_model_features.json"

    pipeline = joblib.load(model_path)
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = [str(f) for f in preprocessor.get_feature_names_out()]

    payload = {
        "source_model": str(model_path),
        "n_features": len(feature_names),
        "features": feature_names,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved {len(feature_names)} transformed model features to {out_path}")


if __name__ == "__main__":
    main()