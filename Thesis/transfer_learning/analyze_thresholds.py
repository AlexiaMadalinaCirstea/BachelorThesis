from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Threshold analysis for transfer-learning prediction outputs."
    )
    parser.add_argument(
        "--predictions_csv",
        required=True,
        help="Path to predictions.csv containing y_true and y_score_attack.",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Directory where threshold-analysis outputs should be written.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
        help="Decision thresholds to evaluate.",
    )
    return parser.parse_args()


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_attack": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_attack": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_attack": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }


def main() -> None:
    args = parse_args()

    predictions_path = Path(args.predictions_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(predictions_path)
    required_cols = {"y_true", "y_score_attack"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"predictions.csv is missing required columns: {missing}")

    y_true = df["y_true"].astype(int)
    rows = []

    for threshold in args.thresholds:
        y_pred = (df["y_score_attack"] >= threshold).astype(int)
        metrics = compute_metrics(y_true, y_pred)
        rows.append(
            {
                "threshold": threshold,
                **metrics,
            }
        )

    summary_df = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
    summary_df.to_csv(out_dir / "threshold_summary.csv", index=False)

    best_macro = summary_df.sort_values("f1_macro", ascending=False).iloc[0].to_dict()
    best_attack = summary_df.sort_values("f1_attack", ascending=False).iloc[0].to_dict()

    with open(out_dir / "best_thresholds.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "best_by_macro_f1": best_macro,
                "best_by_attack_f1": best_attack,
            },
            handle,
            indent=2,
        )

    print(summary_df.to_string(index=False))
    print(f"Saved threshold analysis to: {out_dir}")


if __name__ == "__main__":
    main()
