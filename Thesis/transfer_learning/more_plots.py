from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SEED_DIRECTION_TITLES = {
    "iot23_train->unsw_test": "IoT-23 -> UNSW-NB15",
    "unsw_train->iot23_test": "UNSW-NB15 -> IoT-23",
}

CONDITION_LABELS = {
    "target_only_updated": "Target Only (Updated)",
    "transfer_learning_updated": "Transfer Learning (Updated)",
}

CONDITION_COLORS = {
    "target_only_updated": "#1b5e20",
    "transfer_learning_updated": "#c62828",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate extra transfer-learning figures from existing output folders."
    )
    parser.add_argument(
        "--seed_summary_csv",
        default="transfer_learning/seed_stability/seed_stability_summary.csv",
        help="Path to seed_stability_summary.csv.",
    )
    parser.add_argument(
        "--seed_gain_csv",
        default="transfer_learning/seed_stability/seed_gain_vs_target_only.csv",
        help="Path to seed_gain_vs_target_only.csv.",
    )
    parser.add_argument(
        "--threshold_base_dir",
        default="transfer_learning/threshold_analysis",
        help="Directory containing threshold analysis subfolders.",
    )
    parser.add_argument(
        "--feature_shift_csv",
        default="transfer_learning/final_tables/transfer_learning_feature_shift_top3.csv",
        help="Path to transfer_learning_feature_shift_top3.csv.",
    )
    parser.add_argument(
        "--out_dir",
        default="transfer_learning/more_figures",
        help="Directory where extra figures should be saved.",
    )
    return parser.parse_args()


def plot_seed_stability(summary_df: pd.DataFrame, out_path: Path) -> None:
    directions = list(SEED_DIRECTION_TITLES.keys())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)

    for idx, direction in enumerate(directions):
        ax = axes[idx]
        direction_df = summary_df[summary_df["direction"] == direction].copy()

        for condition in ["target_only_updated", "transfer_learning_updated"]:
            condition_df = direction_df[direction_df["condition"] == condition].sort_values(
                "target_fraction"
            )
            ax.errorbar(
                condition_df["target_fraction"],
                condition_df["mean_f1_macro"],
                yerr=condition_df["std_f1_macro"].fillna(0.0),
                marker="o",
                linewidth=2.0,
                capsize=4,
                color=CONDITION_COLORS[condition],
                label=CONDITION_LABELS[condition],
            )

        ax.set_title(SEED_DIRECTION_TITLES[direction])
        ax.set_xlabel("Target fraction")
        ax.set_ylim(0.40, 0.93)
        ax.grid(alpha=0.25, linestyle="--")

    axes[0].set_ylabel("Macro F1 (mean ± std across seeds)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Seed Stability of Updated Transfer Learning Results", y=1.08, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_seed_gain(gain_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)

    for idx, direction in enumerate(SEED_DIRECTION_TITLES.keys()):
        ax = axes[idx]
        direction_df = gain_df[gain_df["direction"] == direction].sort_values("target_fraction")
        ax.plot(
            direction_df["target_fraction"],
            direction_df["mean_macro_f1_gain_vs_target_only"],
            marker="o",
            linewidth=2.2,
            color="#c62828",
        )
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
        ax.set_title(SEED_DIRECTION_TITLES[direction])
        ax.set_xlabel("Target fraction")
        ax.grid(alpha=0.25, linestyle="--")

    axes[0].set_ylabel("Mean macro F1 gain vs target-only")
    fig.suptitle("Mean Transfer-Learning Gain Across Seeds", y=1.03, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_diagnostics(threshold_base_dir: Path, out_path: Path) -> None:
    folders = [
        ("unsw_to_iot23_target_only_0p05", "Target Only"),
        ("unsw_to_iot23_transfer_learning_0p05", "Transfer Learning"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)

    for idx, (folder_name, title) in enumerate(folders):
        ax = axes[idx]
        threshold_csv = threshold_base_dir / folder_name / "threshold_summary.csv"
        if not threshold_csv.exists():
            ax.set_title(f"{title} (missing)")
            ax.axis("off")
            continue

        df = pd.read_csv(threshold_csv).sort_values("threshold")
        ax.plot(df["threshold"], df["f1_macro"], marker="o", linewidth=2.0, color="#1565c0", label="Macro F1")
        ax.plot(
            df["threshold"],
            df["recall_attack"],
            marker="s",
            linewidth=2.0,
            color="#ef6c00",
            label="Attack Recall",
        )
        ax.set_title(f"UNSW-NB15 -> IoT-23 | {title}")
        ax.set_xlabel("Threshold")
        ax.set_ylim(0.0, 1.02)
        ax.grid(alpha=0.25, linestyle="--")

    axes[0].set_ylabel("Score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Threshold Diagnostics for the Resistant Direction", y=1.08, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_feature_shift(feature_shift_df: pd.DataFrame, out_path: Path) -> None:
    agg = (
        feature_shift_df.groupby(["direction", "feature"], as_index=False)["absolute_shift"]
        .mean()
        .sort_values(["direction", "absolute_shift"], ascending=[True, False])
    )
    top_rows = []
    for direction, group in agg.groupby("direction"):
        top_rows.append(group.head(5))
    top_df = pd.concat(top_rows, ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharex=False)

    for idx, direction in enumerate(sorted(top_df["direction"].unique())):
        ax = axes[idx]
        direction_df = top_df[top_df["direction"] == direction].sort_values("absolute_shift")
        ax.barh(direction_df["feature"], direction_df["absolute_shift"], color="#6a1b9a")
        ax.set_title(direction)
        ax.set_xlabel("Mean absolute importance shift")
        ax.grid(alpha=0.2, linestyle="--", axis="x")

    fig.suptitle("Top Feature Shifts Between Pretraining and Adaptation", y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_summary_df = pd.read_csv(args.seed_summary_csv)
    seed_gain_df = pd.read_csv(args.seed_gain_csv)
    feature_shift_df = pd.read_csv(args.feature_shift_csv)

    plot_seed_stability(seed_summary_df, out_dir / "transfer_learning_seed_stability.png")
    plot_seed_gain(seed_gain_df, out_dir / "transfer_learning_seed_gain.png")
    plot_threshold_diagnostics(Path(args.threshold_base_dir), out_dir / "transfer_learning_threshold_diagnostics.png")
    plot_feature_shift(feature_shift_df, out_dir / "transfer_learning_feature_shift.png")

    print(f"Saved extra transfer-learning figures to: {out_dir}")


if __name__ == "__main__":
    main()
