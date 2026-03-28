from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"feature", "mean_importance", "feature_stability_index", "transfer_utility", "cv_importance"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return df


def scatter_importance_vs_stability(df: pd.DataFrame, title: str, out_path: Path, top_k: int = 8) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.scatter(df["mean_importance"], df["feature_stability_index"], alpha=0.7)

    top_df = df.sort_values("transfer_utility", ascending=False).head(top_k)
    for _, row in top_df.iterrows():
        ax.annotate(
            row["feature"],
            (row["mean_importance"], row["feature_stability_index"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel("Mean normalized importance")
    ax.set_ylabel("Feature Stability Index (FSI)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def bar_top_transferable(df: pd.DataFrame, title: str, out_path: Path, top_k: int = 10) -> None:
    top_df = df.sort_values("transfer_utility", ascending=False).head(top_k).copy()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_df["feature"][::-1], top_df["transfer_utility"][::-1])
    ax.set_xlabel("Transfer Utility")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def bar_top_unstable(df: pd.DataFrame, title: str, out_path: Path, top_k: int = 10) -> None:
    top_df = df.sort_values("cv_importance", ascending=False).head(top_k).copy()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_df["feature"][::-1], top_df["cv_importance"][::-1])
    ax.set_xlabel("Coefficient of Variation (Importance)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate feature stability plots.")
    parser.add_argument("--rf_summary", required=True)
    parser.add_argument("--xgb_summary", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rf_df = load_summary(Path(args.rf_summary))
    xgb_df = load_summary(Path(args.xgb_summary))

    scatter_importance_vs_stability(
        rf_df,
        "Random Forest: Importance vs Stability",
        out_dir / "rf_importance_vs_stability.png",
    )
    scatter_importance_vs_stability(
        xgb_df,
        "XGBoost: Importance vs Stability",
        out_dir / "xgb_importance_vs_stability.png",
    )

    bar_top_transferable(
        rf_df,
        "Random Forest: Top Transferable Features",
        out_dir / "rf_top_transferable_features.png",
    )
    bar_top_transferable(
        xgb_df,
        "XGBoost: Top Transferable Features",
        out_dir / "xgb_top_transferable_features.png",
    )

    bar_top_unstable(
        rf_df,
        "Random Forest: Top Unstable Features",
        out_dir / "rf_top_unstable_features.png",
    )
    bar_top_unstable(
        xgb_df,
        "XGBoost: Top Unstable Features",
        out_dir / "xgb_top_unstable_features.png",
    )

    print(f"Saved feature stability plots to: {out_dir}")


if __name__ == "__main__":
    main()