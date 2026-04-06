import os
import json
import argparse
from typing import Optional, List, Dict

import pandas as pd


def load_metrics_from_root(root_dir: str, model_name: str) -> pd.DataFrame:
    rows: List[Dict] = []

    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Directory not found: {root_dir}")

    for entry in sorted(os.listdir(root_dir)):
        fold_dir = os.path.join(root_dir, entry)

        if not os.path.isdir(fold_dir):
            continue

        metrics_path = os.path.join(fold_dir, "metrics.json")
        if not os.path.isfile(metrics_path):
            print(f"[WARN] Missing metrics.json in {fold_dir}")
            continue

        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        metrics["model"] = model_name
        metrics["fold_name"] = entry
        rows.append(metrics)

    if not rows:
        raise ValueError(f"No fold metrics found in {root_dir}")

    df = pd.DataFrame(rows)

    preferred_order = [
        "model",
        "held_out_attack",
        "fold_name",
        "train_size",
        "test_size",
        "held_out_attack_support",
        "accuracy",
        "macro_f1",
        "binary_f1",
        "precision_attack",
        "recall_attack",
        "precision_benign",
        "recall_benign",
        "confusion_matrix",
    ]
    cols = [c for c in preferred_order if c in df.columns] + [c for c in df.columns if c not in preferred_order]
    return df[cols]


def summarize_model(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    metric_cols = [
        "accuracy",
        "macro_f1",
        "binary_f1",
        "precision_attack",
        "recall_attack",
        "precision_benign",
        "recall_benign",
    ]

    summary = []
    for col in metric_cols:
        if col not in df.columns:
            continue
        summary.append({
            "model": model_name,
            "metric": col,
            "mean": df[col].mean(),
            "std": df[col].std(),
            "min": df[col].min(),
            "max": df[col].max(),
        })

    return pd.DataFrame(summary)


def best_worst_table(df: pd.DataFrame, model_name: str, sort_metric: str = "macro_f1") -> pd.DataFrame:
    if sort_metric not in df.columns:
        raise ValueError(f"Metric '{sort_metric}' not found in DataFrame")

    sorted_df = df.sort_values(sort_metric, ascending=False).reset_index(drop=True)

    best_row = sorted_df.iloc[0].copy()
    worst_row = sorted_df.iloc[-1].copy()

    out = pd.DataFrame([best_row, worst_row])
    out.insert(1, "rank_type", ["best", "worst"])

    # only add model column if it doesn't exist
    if "model" not in out.columns:
        out.insert(0, "model", model_name)

    return out

def compare_models(rf_df: pd.DataFrame, xgb_df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "held_out_attack",
        "accuracy",
        "macro_f1",
        "binary_f1",
        "precision_attack",
        "recall_attack",
        "precision_benign",
        "recall_benign",
    ]

    rf_small = rf_df[[c for c in keep_cols if c in rf_df.columns]].copy()
    xgb_small = xgb_df[[c for c in keep_cols if c in xgb_df.columns]].copy()

    rf_small = rf_small.rename(columns={c: f"{c}_rf" for c in rf_small.columns if c != "held_out_attack"})
    xgb_small = xgb_small.rename(columns={c: f"{c}_xgb" for c in xgb_small.columns if c != "held_out_attack"})

    merged = pd.merge(rf_small, xgb_small, on="held_out_attack", how="inner")

    for metric in ["accuracy", "macro_f1", "binary_f1", "precision_attack", "recall_attack", "precision_benign", "recall_benign"]:
        rf_col = f"{metric}_rf"
        xgb_col = f"{metric}_xgb"
        if rf_col in merged.columns and xgb_col in merged.columns:
            merged[f"{metric}_rf_minus_xgb"] = merged[rf_col] - merged[xgb_col]

    return merged.sort_values("macro_f1_rf_minus_xgb", ascending=False)


def save_text_report(
    out_path: str,
    rf_df: Optional[pd.DataFrame],
    xgb_df: Optional[pd.DataFrame],
    rf_summary: Optional[pd.DataFrame],
    xgb_summary: Optional[pd.DataFrame],
    comparison_df: Optional[pd.DataFrame],
) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("UNSW-NB15 Leave-One-Attack-Type-Out Analysis\n")
        f.write("=" * 50 + "\n\n")

        if rf_df is not None:
            f.write("Random Forest\n")
            f.write("-" * 20 + "\n")
            f.write(f"Folds: {len(rf_df)}\n")
            f.write(f"Best held-out attack by macro_f1: {rf_df.sort_values('macro_f1', ascending=False).iloc[0]['held_out_attack']}\n")
            f.write(f"Worst held-out attack by macro_f1: {rf_df.sort_values('macro_f1', ascending=True).iloc[0]['held_out_attack']}\n\n")
            if rf_summary is not None:
                f.write(rf_summary.to_string(index=False))
                f.write("\n\n")

        if xgb_df is not None:
            f.write("XGBoost\n")
            f.write("-" * 20 + "\n")
            f.write(f"Folds: {len(xgb_df)}\n")
            f.write(f"Best held-out attack by macro_f1: {xgb_df.sort_values('macro_f1', ascending=False).iloc[0]['held_out_attack']}\n")
            f.write(f"Worst held-out attack by macro_f1: {xgb_df.sort_values('macro_f1', ascending=True).iloc[0]['held_out_attack']}\n\n")
            if xgb_summary is not None:
                f.write(xgb_summary.to_string(index=False))
                f.write("\n\n")

        if comparison_df is not None:
            f.write("RF vs XGB comparison by held-out attack\n")
            f.write("-" * 40 + "\n")
            cols_to_show = [
                c for c in [
                    "held_out_attack",
                    "macro_f1_rf",
                    "macro_f1_xgb",
                    "macro_f1_rf_minus_xgb",
                    "binary_f1_rf",
                    "binary_f1_xgb",
                    "binary_f1_rf_minus_xgb",
                ] if c in comparison_df.columns
            ]
            f.write(comparison_df[cols_to_show].to_string(index=False))
            f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze leave-one-attack-type-out results for UNSW-NB15")
    parser.add_argument("--rf_dir", default=None, help="Root folder for RF results")
    parser.add_argument("--xgb_dir", default=None, help="Root folder for XGB results")
    parser.add_argument("--out_dir", required=True, help="Directory to save analysis outputs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    rf_df = None
    xgb_df = None
    rf_summary = None
    xgb_summary = None
    comparison_df = None

    if args.rf_dir:
        rf_df = load_metrics_from_root(args.rf_dir, "rf")
        rf_df.to_csv(os.path.join(args.out_dir, "rf_folds_detailed.csv"), index=False)
        rf_summary = summarize_model(rf_df, "rf")
        rf_summary.to_csv(os.path.join(args.out_dir, "rf_summary_stats.csv"), index=False)
        best_worst_table(rf_df, "rf").to_csv(os.path.join(args.out_dir, "rf_best_worst.csv"), index=False)

    if args.xgb_dir:
        xgb_df = load_metrics_from_root(args.xgb_dir, "xgb")
        xgb_df.to_csv(os.path.join(args.out_dir, "xgb_folds_detailed.csv"), index=False)
        xgb_summary = summarize_model(xgb_df, "xgb")
        xgb_summary.to_csv(os.path.join(args.out_dir, "xgb_summary_stats.csv"), index=False)
        best_worst_table(xgb_df, "xgb").to_csv(os.path.join(args.out_dir, "xgb_best_worst.csv"), index=False)

    if rf_df is not None and xgb_df is not None:
        comparison_df = compare_models(rf_df, xgb_df)
        comparison_df.to_csv(os.path.join(args.out_dir, "rf_vs_xgb_comparison.csv"), index=False)

    save_text_report(
        out_path=os.path.join(args.out_dir, "analysis_report.txt"),
        rf_df=rf_df,
        xgb_df=xgb_df,
        rf_summary=rf_summary,
        xgb_summary=xgb_summary,
        comparison_df=comparison_df,
    )

    print(f"Saved analysis outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()