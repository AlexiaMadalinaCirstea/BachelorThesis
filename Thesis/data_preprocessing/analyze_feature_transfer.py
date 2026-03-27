from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


def infer_model_type_from_dir(path: Path) -> str:
    name = path.name.lower()
    if "rf" in name:
        return "rf"
    if "xgb" in name:
        return "xgb"
    raise ValueError(f"Could not infer model type from directory name: {path}")


def get_feature_names_from_pipeline(pipeline) -> list[str]:
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    return [str(f) for f in feature_names]


def get_importances_from_pipeline(pipeline, model_type: str) -> np.ndarray:
    model = pipeline.named_steps["model"]

    if not hasattr(model, "feature_importances_"):
        raise AttributeError(f"Model does not expose feature_importances_: {type(model)}")

    return np.asarray(model.feature_importances_, dtype=float)


def parse_fold_name(path: Path) -> str:
    return path.parent.name


def collect_model_importances(root_dir: Path, model_type: str) -> pd.DataFrame:
    model_paths = sorted(root_dir.glob("loso_*/model.joblib"))
    if not model_paths:
        raise FileNotFoundError(f"No model.joblib files found under {root_dir}")

    rows = []

    for model_path in model_paths:
        pipeline = joblib.load(model_path)
        feature_names = get_feature_names_from_pipeline(pipeline)
        importances = get_importances_from_pipeline(pipeline, model_type=model_type)

        if len(feature_names) != len(importances):
            raise ValueError(
                f"Feature name / importance length mismatch for {model_path}: "
                f"{len(feature_names)} vs {len(importances)}"
            )

        fold_name = parse_fold_name(model_path)

        for feature, importance in zip(feature_names, importances):
            rows.append(
                {
                    "fold_name": fold_name,
                    "feature": feature,
                    "importance": float(importance),
                }
            )

    return pd.DataFrame(rows)


def summarize_importances(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("feature", as_index=False)
        .agg(
            mean_importance=("importance", "mean"),
            std_importance=("importance", "std"),
            min_importance=("importance", "min"),
            max_importance=("importance", "max"),
            n_folds=("importance", "count"),
        )
    )

    summary["std_importance"] = summary["std_importance"].fillna(0.0)
    summary["cv_importance"] = summary["std_importance"] / (summary["mean_importance"] + 1e-12)

    # high mean + low std = stable
    summary["stability_score"] = summary["mean_importance"] / (summary["std_importance"] + 1e-12)

    summary = summary.sort_values(
        ["mean_importance", "stability_score"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return summary


def build_cross_model_comparison(rf_summary: pd.DataFrame, xgb_summary: pd.DataFrame) -> pd.DataFrame:
    merged = rf_summary.merge(
        xgb_summary,
        on="feature",
        how="outer",
        suffixes=("_rf", "_xgb"),
    ).fillna(0.0)

    merged["mean_gap_abs"] = (merged["mean_importance_rf"] - merged["mean_importance_xgb"]).abs()
    merged["shared_importance"] = merged[["mean_importance_rf", "mean_importance_xgb"]].min(axis=1)

    merged["shared_stability"] = merged[["stability_score_rf", "stability_score_xgb"]].min(axis=1)

    merged = merged.sort_values(
        ["shared_importance", "shared_stability"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return merged


def save_ranked_views(summary: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    summary.to_csv(out_dir / f"{prefix}_feature_importance_summary.csv", index=False)

    summary.sort_values("mean_importance", ascending=False).head(30).to_csv(
        out_dir / f"{prefix}_top30_by_mean_importance.csv", index=False
    )
    summary.sort_values("stability_score", ascending=False).head(30).to_csv(
        out_dir / f"{prefix}_top30_stable_features.csv", index=False
    )
    summary.sort_values("cv_importance", ascending=False).head(30).to_csv(
        out_dir / f"{prefix}_top30_unstable_features.csv", index=False
    )


def print_brief(summary: pd.DataFrame, label: str) -> None:
    top_stable = summary.sort_values(
        ["mean_importance", "stability_score"],
        ascending=[False, False],
    ).head(10)[["feature", "mean_importance", "std_importance", "stability_score"]]

    top_unstable = summary.sort_values(
        ["mean_importance", "cv_importance"],
        ascending=[False, False],
    ).head(10)[["feature", "mean_importance", "std_importance", "cv_importance"]]

    print(f"\n=== {label}: top stable features ===")
    print(top_stable.to_string(index=False))

    print(f"\n=== {label}: top high-importance / unstable features ===")
    print(top_unstable.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze feature transfer across LOSO-trained models")
    parser.add_argument("--rf_dir", required=True, help="Path to rf_loso directory")
    parser.add_argument("--xgb_dir", required=True, help="Path to xgb_loso directory")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    args = parser.parse_args()

    rf_dir = Path(args.rf_dir)
    xgb_dir = Path(args.xgb_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rf_df = collect_model_importances(rf_dir, model_type="rf")
    xgb_df = collect_model_importances(xgb_dir, model_type="xgb")

    rf_df.to_csv(out_dir / "rf_fold_feature_importances_long.csv", index=False)
    xgb_df.to_csv(out_dir / "xgb_fold_feature_importances_long.csv", index=False)

    rf_summary = summarize_importances(rf_df)
    xgb_summary = summarize_importances(xgb_df)

    save_ranked_views(rf_summary, out_dir, prefix="rf")
    save_ranked_views(xgb_summary, out_dir, prefix="xgb")

    comparison = build_cross_model_comparison(rf_summary, xgb_summary)
    comparison.to_csv(out_dir / "rf_xgb_feature_comparison.csv", index=False)

    print_brief(rf_summary, "RF")
    print_brief(xgb_summary, "XGB")

    print("\nSaved outputs to:", out_dir)


if __name__ == "__main__":
    main()