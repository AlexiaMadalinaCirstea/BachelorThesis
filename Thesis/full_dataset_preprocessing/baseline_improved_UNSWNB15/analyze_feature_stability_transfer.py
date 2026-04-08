from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


EPS = 1e-12
PRESENCE_THRESHOLD = 1e-8


def infer_model_type_from_dir(path: Path) -> str:
    name = path.name.lower()
    if "rf" in name:
        return "rf"
    if "xgb" in name:
        return "xgb"
    raise ValueError(f"Could not infer model type from directory name: {path}")


def extract_feature_importances_from_pipeline(model_path: Path) -> pd.DataFrame:
    pipeline = joblib.load(model_path)
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    importances = getattr(model, "feature_importances_", None)

    if importances is None:
        raise AttributeError(f"Model does not expose feature_importances_: {type(model)}")
    if len(feature_names) != len(importances):
        raise ValueError(
            f"Feature name / importance length mismatch for {model_path}: "
            f"{len(feature_names)} vs {len(importances)}"
        )

    return pd.DataFrame({
        "feature": [str(name) for name in feature_names],
        "importance": np.asarray(importances, dtype=float),
    })


def collect_fold_importances(root_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    if not root_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {root_dir}")

    for fold_dir in sorted(root_dir.iterdir()):
        if not fold_dir.is_dir():
            continue

        csv_path = fold_dir / "fold_feature_importances.csv"
        model_path = fold_dir / "model.joblib"

        if csv_path.exists():
            fold_df = pd.read_csv(csv_path)
        elif model_path.exists():
            fold_df = extract_feature_importances_from_pipeline(model_path)
        else:
            continue

        required = {"feature", "importance"}
        missing = required - set(fold_df.columns)
        if missing:
            raise ValueError(f"{csv_path} missing required columns: {missing}")

        fold_name = fold_dir.name
        for row in fold_df.itertuples(index=False):
            rows.append(
                {
                    "fold_name": fold_name,
                    "feature": str(row.feature),
                    "importance": float(row.importance),
                }
            )

    if not rows:
        raise FileNotFoundError(
            f"No fold_feature_importances.csv or model.joblib files found under {root_dir}"
        )

    return pd.DataFrame(rows)


def normalize_and_rank(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["importance_norm"] = df.groupby("fold_name")["importance"].transform(
        lambda x: x / (x.sum() + EPS)
    )
    df["rank"] = df.groupby("fold_name")["importance_norm"].rank(
        ascending=False, method="first"
    )
    df["rank_norm"] = df.groupby("fold_name")["rank"].transform(
        lambda x: x / x.max()
    )
    df["is_present"] = (df["importance_norm"] > PRESENCE_THRESHOLD).astype(float)
    return df


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
    summary["cv_importance"] = summary["std_importance"] / (summary["mean_importance"] + EPS)
    summary["stability_presence"] = summary["presence_rate"]
    summary["stability_variance"] = 1.0 / (1.0 + summary["cv_importance"])
    summary["stability_rank"] = 1.0 / (1.0 + summary["std_rank"])
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

    return summary.sort_values(
        ["transfer_utility", "feature_stability_index", "mean_importance"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def save_ranked_views(summary: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    summary.to_csv(out_dir / f"{prefix}_feature_stability_summary.csv", index=False)
    summary.sort_values("mean_importance", ascending=False).head(30).to_csv(
        out_dir / f"{prefix}_top30_by_mean_importance.csv", index=False
    )
    summary.sort_values("feature_stability_index", ascending=False).head(30).to_csv(
        out_dir / f"{prefix}_top30_stable_features.csv", index=False
    )
    summary.sort_values("transfer_utility", ascending=False).head(30).to_csv(
        out_dir / f"{prefix}_top30_transferable_features.csv", index=False
    )
    summary.sort_values("cv_importance", ascending=False).head(30).to_csv(
        out_dir / f"{prefix}_top30_unstable_features.csv", index=False
    )


def build_cross_model_comparison(
    rf_summary: pd.DataFrame,
    xgb_summary: pd.DataFrame,
) -> pd.DataFrame:
    merged = rf_summary.merge(
        xgb_summary,
        on="feature",
        how="outer",
        suffixes=("_rf", "_xgb"),
    ).fillna(0.0)

    merged["shared_importance"] = merged[["mean_importance_rf", "mean_importance_xgb"]].min(axis=1)
    merged["shared_stability"] = merged[
        ["feature_stability_index_rf", "feature_stability_index_xgb"]
    ].min(axis=1)
    merged["shared_transfer_utility"] = merged[
        ["transfer_utility_rf", "transfer_utility_xgb"]
    ].min(axis=1)
    merged["mean_gap_abs"] = (
        merged["mean_importance_rf"] - merged["mean_importance_xgb"]
    ).abs()

    return merged.sort_values(
        ["shared_transfer_utility", "shared_stability", "shared_importance"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def print_insights(summary: pd.DataFrame, label: str) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze feature stability and transfer on UNSW-NB15 leave-one-attack-type-out runs"
    )
    parser.add_argument("--rf_dir", required=True, help="Root folder for RF fold outputs")
    parser.add_argument("--xgb_dir", required=True, help="Root folder for XGB fold outputs")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    args = parser.parse_args()

    rf_dir = Path(args.rf_dir)
    xgb_dir = Path(args.xgb_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Detected model type for {rf_dir.name}: {infer_model_type_from_dir(rf_dir)}")
    print(f"Detected model type for {xgb_dir.name}: {infer_model_type_from_dir(xgb_dir)}")

    rf_long = collect_fold_importances(rf_dir)
    xgb_long = collect_fold_importances(xgb_dir)

    rf_long.to_csv(out_dir / "rf_fold_feature_importances_long.csv", index=False)
    xgb_long.to_csv(out_dir / "xgb_fold_feature_importances_long.csv", index=False)

    rf_summary = compute_feature_stability(normalize_and_rank(rf_long))
    xgb_summary = compute_feature_stability(normalize_and_rank(xgb_long))

    save_ranked_views(rf_summary, out_dir, "rf")
    save_ranked_views(xgb_summary, out_dir, "xgb")

    comparison = build_cross_model_comparison(rf_summary, xgb_summary)
    comparison.to_csv(out_dir / "rf_xgb_feature_comparison.csv", index=False)

    print_insights(rf_summary, "UNSW RF")
    print_insights(xgb_summary, "UNSW XGB")
    print(f"\nSaved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
