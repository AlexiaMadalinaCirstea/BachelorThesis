from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_NON_FEATURE_COLS = {
    "id",
    "attack_cat",
    "label",
    "split",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export raw UNSW-NB15 feature columns from a processed or official CSV split."
    )
    parser.add_argument(
        "--train_path",
        default="Datasets/UNSW-NB15/UNSW-NB15 dataset/CSV Files/Training and Testing Sets/UNSW_NB15_training-set.csv",
        help="Path to the UNSW-NB15 train CSV file.",
    )
    parser.add_argument(
        "--out_dir",
        default="feature_alignment",
        help="Directory where the exported feature JSON will be written.",
    )
    parser.add_argument(
        "--out_name",
        default="unsw_features.json",
        help="Output JSON filename.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_path = Path(args.train_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out_name

    df = pd.read_csv(train_path, nrows=1)
    columns = list(df.columns)

    feature_cols = [col for col in columns if col not in DEFAULT_NON_FEATURE_COLS]

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
