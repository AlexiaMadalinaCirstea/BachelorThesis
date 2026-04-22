from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRICS = [
    ("f1_attack", "Attack F1"),
    ("recall_attack", "Attack Recall"),
    ("f1_macro", "Macro F1"),
]

CONDITION_COLORS = {
    "target_only_updated": "#1b5e20",
    "transfer_learning_updated": "#c62828",
}

CONDITION_LABELS = {
    "target_only_updated": "Target Only (Updated)",
    "transfer_learning_updated": "Transfer Learning (Updated)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot transfer-learning updated recipe results."
    )
    parser.add_argument(
        "--summary_csv",
        default="transfer_learning/outputs_updated_recipe/updated_recipe_summary.csv",
        help="Path to updated_recipe_summary.csv",
    )
    parser.add_argument(
        "--out_dir",
        default="transfer_learning/figures",
        help="Directory for generated figures.",
    )
    return parser.parse_args()


def direction_title(direction: str) -> str:
    mapping = {
        "iot23_train->unsw_test": "IoT-23 -> UNSW-NB15",
        "unsw_train->iot23_test": "UNSW-NB15 -> IoT-23",
    }
    return mapping.get(direction, direction)


def plot_metric_grid(df: pd.DataFrame, out_path: Path) -> None:
    directions = [
        "iot23_train->unsw_test",
        "unsw_train->iot23_test",
    ]
    conditions = [
        "target_only_updated",
        "transfer_learning_updated",
    ]

    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(16, 8),
        sharex=True,
        sharey="col",
    )

    for row_idx, direction in enumerate(directions):
        direction_df = df[df["direction"] == direction].copy()

        for col_idx, (metric_col, metric_label) in enumerate(METRICS):
            ax = axes[row_idx, col_idx]

            for condition in conditions:
                condition_df = direction_df[direction_df["condition"] == condition].copy()
                if condition_df.empty:
                    continue

                condition_df = condition_df.sort_values("target_fraction")
                ax.plot(
                    condition_df["target_fraction"],
                    condition_df[metric_col],
                    marker="o",
                    linewidth=2.2,
                    markersize=6,
                    color=CONDITION_COLORS[condition],
                    label=CONDITION_LABELS[condition],
                )

            ax.set_title(f"{direction_title(direction)} | {metric_label}")
            ax.set_ylim(0.0, 1.0)
            ax.grid(alpha=0.25, linestyle="--")

            if row_idx == len(directions) - 1:
                ax.set_xlabel("Target fraction")
            if col_idx == 0:
                ax.set_ylabel("Score")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Transfer Learning Updated Recipe: Target Only vs Transfer Learning", y=1.06, fontsize=15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_improvement_grid(df: pd.DataFrame, out_path: Path) -> None:
    conditions = ["target_only_updated", "transfer_learning_updated"]
    directions = [
        "iot23_train->unsw_test",
        "unsw_train->iot23_test",
    ]

    base_df = df[df["condition"] == "target_only_updated"][["direction", "target_fraction", "f1_macro"]].copy()
    base_df = base_df.rename(columns={"f1_macro": "target_only_f1_macro"})

    merged = df.merge(base_df, on=["direction", "target_fraction"], how="left")
    merged["macro_f1_gain_vs_target_only"] = merged["f1_macro"] - merged["target_only_f1_macro"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)

    for idx, direction in enumerate(directions):
        ax = axes[idx]
        direction_df = merged[merged["direction"] == direction].copy()

        for condition in conditions:
            condition_df = direction_df[direction_df["condition"] == condition].copy()
            if condition_df.empty:
                continue
            condition_df = condition_df.sort_values("target_fraction")
            ax.plot(
                condition_df["target_fraction"],
                condition_df["macro_f1_gain_vs_target_only"],
                marker="o",
                linewidth=2.2,
                markersize=6,
                color=CONDITION_COLORS[condition],
                label=CONDITION_LABELS[condition],
            )

        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
        ax.set_title(direction_title(direction))
        ax.set_xlabel("Target fraction")
        ax.grid(alpha=0.25, linestyle="--")

    axes[0].set_ylabel("Macro F1 gain vs target-only")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Updated Recipe: Gain Relative to Target-Only Training", y=1.08, fontsize=15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    summary_path = Path(args.summary_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_path)

    required_cols = {"direction", "condition", "target_fraction", "f1_attack", "recall_attack", "f1_macro"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Summary CSV is missing required columns: {missing}")

    df = df[df["condition"].isin(CONDITION_LABELS)].copy()
    df["target_fraction"] = pd.to_numeric(df["target_fraction"], errors="coerce")

    plot_metric_grid(
        df=df,
        out_path=out_dir / "transfer_learning_updated_recipe_progression.png",
    )
    plot_improvement_grid(
        df=df,
        out_path=out_dir / "transfer_learning_updated_recipe_gain_vs_target_only.png",
    )

    print(f"Saved figures to: {out_dir}")


if __name__ == "__main__":
    main()
