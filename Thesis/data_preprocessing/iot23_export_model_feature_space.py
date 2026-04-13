from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export transformed IoT-23 model feature names from a trained pipeline."
    )
    parser.add_argument(
        "--model_path",
        default="Datasets/IoT23/processed_test_sample/iot23/rf_baseline/rf_model.joblib",
        help="Path to the trained IoT-23 model pipeline.",
    )
    parser.add_argument(
        "--out_dir",
        default="feature_alignment",
        help="Directory where the exported model feature JSON will be written.",
    )
    parser.add_argument(
        "--out_name",
        default="iot23_model_features.json",
        help="Output JSON filename.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_path = Path(args.model_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / args.out_name

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
