from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BRANCHES = ["tabular", "temporal", "prototype"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze hybrid full-model run outputs.")
    parser.add_argument(
        "--runs_dir",
        default="early_detection/hybrid",
        help="Directory containing hybrid run folders.",
    )
    parser.add_argument(
        "--out_dir",
        default="early_detection/hybrid/hybrid_analyzer",
        help="Directory for analyzer outputs.",
    )
    return parser.parse_args()


def collect_fraction_summaries(runs_dir: Path) -> pd.DataFrame:
    rows = []
    for csv_path in runs_dir.rglob("overall_fraction_summary.csv"):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if df.empty:
            continue
        if "direction" not in df.columns:
            continue
        df["run_dir"] = str(csv_path.parent)
        rows.append(df)
    if not rows:
        raise ValueError(f"No hybrid overall_fraction_summary.csv files found under {runs_dir}")
    return pd.concat(rows, ignore_index=True)


def make_branch_weight_plot(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, split_name in zip(axes, ["val", "test"]):
        split_df = df[df["split"] == split_name]
        if split_df.empty:
            ax.set_visible(False)
            continue
        for branch in BRANCHES:
            col = f"mean_{branch}_weight"
            if col not in split_df.columns:
                continue
            grouped = split_df.groupby("fraction", as_index=False)[col].mean()
            ax.plot(grouped["fraction"], grouped[col], marker="o", label=branch)
        ax.set_title(f"{split_name} mean branch weight")
        ax.set_xlabel("Fraction")
        ax.set_ylabel("Weight")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def make_branch_winrate_plot(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, split_name in zip(axes, ["val", "test"]):
        split_df = df[df["split"] == split_name]
        if split_df.empty:
            ax.set_visible(False)
            continue
        for branch in BRANCHES:
            col = f"{branch}_branch_win_rate"
            if col not in split_df.columns:
                continue
            grouped = split_df.groupby("fraction", as_index=False)[col].mean()
            ax.plot(grouped["fraction"], grouped[col], marker="o", label=branch)
        ax.set_title(f"{split_name} branch dominance")
        ax.set_xlabel("Fraction")
        ax.set_ylabel("Win rate")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def make_metric_plot(df: pd.DataFrame, metric: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    test_df = df[df["split"] == "test"].copy()
    if test_df.empty or metric not in test_df.columns:
        plt.close(fig)
        return

    group_cols = ["direction"]
    if "ablation_name" in test_df.columns:
        group_cols.append("ablation_name")
    grouped = test_df.groupby(group_cols + ["fraction"], as_index=False)[metric].mean()
    for key, subset in grouped.groupby(group_cols, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        label = " | ".join(str(x) for x in key)
        ax.plot(subset["fraction"], subset[metric], marker="o", label=label)
    ax.set_title(f"Test {metric} vs fraction")
    ax.set_xlabel("Fraction")
    ax.set_ylabel(metric)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary_df = collect_fraction_summaries(runs_dir)
    summary_df.to_csv(out_dir / "all_fraction_summaries.csv", index=False)

    agg_cols = [
        col
        for col in summary_df.columns
        if col
        not in {
            "run_dir",
            "source_train_rows",
            "target_val_rows_config",
            "target_test_rows_config",
            "n_aligned_features",
            "rows_evaluated",
            "n_scenarios",
            "n_attack_categories",
            "attack_support",
            "false_negatives",
            "false_positives",
            "true_negatives",
            "true_positives",
        }
    ]
    group_cols = [col for col in ["direction", "ablation_name", "split", "fraction"] if col in summary_df.columns]
    numeric_cols = [col for col in agg_cols if col not in group_cols and pd.api.types.is_numeric_dtype(summary_df[col])]
    aggregated = summary_df.groupby(group_cols, as_index=False)[numeric_cols].mean()
    aggregated.to_csv(out_dir / "aggregated_fraction_summary.csv", index=False)

    make_branch_weight_plot(aggregated, plots_dir / "branch_weights_vs_fraction.png")
    make_branch_winrate_plot(aggregated, plots_dir / "branch_winrates_vs_fraction.png")
    make_metric_plot(aggregated, "f1_attack", plots_dir / "f1_attack_vs_fraction.png")
    make_metric_plot(aggregated, "recall_attack", plots_dir / "recall_attack_vs_fraction.png")
    make_metric_plot(aggregated, "mean_branch_agreement", plots_dir / "branch_agreement_vs_fraction.png")
    make_metric_plot(aggregated, "mean_prototype_margin", plots_dir / "prototype_margin_vs_fraction.png")


if __name__ == "__main__":
    main()
