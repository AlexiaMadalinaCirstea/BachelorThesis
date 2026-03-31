# EXPORTS RAW FEATURES 

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def main() -> None:
    train_path = Path("Datasets/IoT23/processed_test_sample/iot23/train.parquet")
    out_dir = Path("feature_alignment")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "iot23_features.json"

    df = pd.read_parquet(train_path)

    # Adjust this set if your train file contains other metadata columns
    non_feature_cols = {
        "label",
        "label_binary",
        "label_multiphase",
        "scenario_id",
        "scenario",
        "split",
        "ts",
        "uid",
    }

    feature_cols = [col for col in df.columns if col not in non_feature_cols]

    payload = {
        "source": str(train_path),
        "n_features": len(feature_cols),
        "features": feature_cols,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved {len(feature_cols)} features to {out_path}")


if __name__ == "__main__":
    main()