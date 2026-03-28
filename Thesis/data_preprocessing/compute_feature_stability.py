from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EPS = 1e-12
PRESENCE_THRESHOLD = 1e-8


# load existing long form data 


def load_long_importance_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required_cols = {"fold_name", "feature", "importance"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"{path} must contain columns: {required_cols}"
        )

    return df



# normalization + ranking


def normalize_and_rank(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # normalize per fold
    df["importance_norm"] = df.groupby("fold_name")["importance"].transform(
        lambda x: x / (x.sum() + EPS)
    )

    # compute ranks (1 = most important)
    df["rank"] = df.groupby("fold_name")["importance_norm"].rank(
        ascending=False, method="first"
    )

    # normalize rank to (0,1]
    df["rank_norm"] = df.groupby("fold_name")["rank"].transform(
        lambda x: x / x.max()
    )

    # presence
    df["is_present"] = (df["importance_norm"] > PRESENCE_THRESHOLD).astype(float)

    return df


# feature stability metric

def compute_feature_stability(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("feature", as_index=False)
        .agg(
            mean_importance=("importance_norm", "mean"),
            std_importance=("importance_norm", "std"),
            min_importance=("importance_norm", "min"),
            max_importance=("importance_norm", "max"),
            n_folds=("importance_norm", "count"),
            mean_rank=("rank_norm", "mean"),
            std_rank=("rank_norm", "std"),
            presence_rate=("is_present", "mean"),
        )
    )

    summary["std_importance"] = summary["std_importance"].fillna(0.0)
    summary["std_rank"] = summary["std_rank"].fillna(0.0)

    # coefficient of variation
    summary["cv_importance"] = (
        summary["std_importance"] / (summary["mean_importance"] + EPS)
    )

    # feature stability components

    summary["stability_presence"] = summary["presence_rate"]
    summary["stability_variance"] = 1.0 / (1.0 + summary["cv_importance"])
    summary["stability_rank"] = 1.0 / (1.0 + summary["std_rank"])

   
    # my final metrics
    
    summary["feature_stability_index"] = (
        summary["stability_presence"]
        * summary["stability_variance"]
        * summary["stability_rank"]
    )

    summary["transfer_utility"] = (
        summary["mean_importance"] * summary["feature_stability_index"]
    )

    summary["stability_score_old"] = (
        summary["mean_importance"] / (summary["std_importance"] + EPS)
    )

    summary = summary.sort_values(
        ["transfer_utility", "feature_stability_index", "mean_importance"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return summary



def save_outputs(summary: pd.DataFrame, out_dir: Path, prefix: str):
    summary.to_csv(out_dir / f"{prefix}_feature_stability_summary.csv", index=False)

    # top important
    summary.sort_values("mean_importance", ascending=False).head(30).to_csv(
        out_dir / f"{prefix}_top30_by_mean_importance.csv",
        index=False,
    )

    # top stable
    summary.sort_values("feature_stability_index", ascending=False).head(30).to_csv(
        out_dir / f"{prefix}_top30_stable_features.csv",
        index=False,
    )

    # top transferable (NEW)
    summary.sort_values("transfer_utility", ascending=False).head(30).to_csv(
        out_dir / f"{prefix}_top30_transferable_features.csv",
        index=False,
    )

    # unstable
    summary.sort_values("cv_importance", ascending=False).head(30).to_csv(
        out_dir / f"{prefix}_top30_unstable_features.csv",
        index=False,
    )


def print_insights(summary: pd.DataFrame, label: str):
    print(f"\n=== {label} ===")

    print("\nTop stable features:")
    print(
        summary.sort_values("feature_stability_index", ascending=False)
        .head(5)[["feature", "feature_stability_index", "mean_importance"]]
    )

    print("\nTop transferable features:")
    print(
        summary.sort_values("transfer_utility", ascending=False)
        .head(5)[["feature", "transfer_utility", "feature_stability_index"]]
    )

    print("\nTop unstable features:")
    print(
        summary.sort_values("cv_importance", ascending=False)
        .head(5)[["feature", "cv_importance", "mean_importance"]]
    )



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rf_long", required=True)
    parser.add_argument("--xgb_long", required=True)
    parser.add_argument("--out_dir", required=True)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rf_df = load_long_importance_csv(Path(args.rf_long))
    xgb_df = load_long_importance_csv(Path(args.xgb_long))

    rf_df = normalize_and_rank(rf_df)
    xgb_df = normalize_and_rank(xgb_df)

    rf_summary = compute_feature_stability(rf_df)
    xgb_summary = compute_feature_stability(xgb_df)

    save_outputs(rf_summary, out_dir, "rf")
    save_outputs(xgb_summary, out_dir, "xgb")

    print_insights(rf_summary, "Random Forest")
    print_insights(xgb_summary, "XGBoost")

    print(f"\nSaved results to: {out_dir}")


if __name__ == "__main__":
    main()