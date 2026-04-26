from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter


DIRECTION_LABELS = {
    "iot23_to_unsw": "IoT-23 -> UNSW-NB15",
    "unsw_to_iot23": "UNSW-NB15 -> IoT-23",
}

DIRECTION_COLORS = {
    "iot23_to_unsw": "#b33a3a",
    "unsw_to_iot23": "#1f6f8b",
}

TRANSFER_LABEL_COLORS = {
    "positive": "#2e7d32",
    "neutral": "#9e9e9e",
    "negative": "#c62828",
}

TRANSFER_LABEL_ORDER = ["Positive", "Neutral", "Negative"]

EXPORT_DPI = 600


def apply_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": EXPORT_DPI,
            "axes.titlesize": 18,
            "axes.labelsize": 15,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )


def save_figure(fig: plt.Figure, out_path: Path) -> None:
    fig.savefig(out_path, dpi=EXPORT_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def format_fraction_ticks(ax: plt.Axes) -> None:
    ax.set_xticks([0.10, 0.50, 1.00])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot combined direction-analysis figures for the transfer-learning hypothesis."
    )
    parser.add_argument(
        "--analysis_dir",
        default="transfer_learning/hypothesis/combined_direction_analysis",
        help="Directory containing combined direction-analysis CSV outputs.",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Directory for plot outputs. Defaults to <analysis_dir>/figures.",
    )
    return parser.parse_args()


def prettify_direction(value: str) -> str:
    return DIRECTION_LABELS.get(value, value)


def add_direction_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["direction_label"] = df["pair_family"].map(prettify_direction)
    return df


def plot_direction_fraction_gain(fraction_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 6.5), constrained_layout=True)
    for pair_family, group_df in fraction_df.groupby("pair_family", dropna=False):
        group_df = group_df.sort_values("target_fraction")
        ax.plot(
            group_df["target_fraction"],
            group_df["mean_primary_gain"],
            marker="o",
            linewidth=2.4,
            markersize=7,
            color=DIRECTION_COLORS.get(pair_family, "#333333"),
            label=prettify_direction(pair_family),
        )

    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Target fraction")
    ax.set_ylabel("Mean transfer gain (F1 attack)")
    ax.set_title("Mean Transfer Gain by Direction and Target Fraction")
    ax.grid(alpha=0.25, linestyle="--")
    format_fraction_ticks(ax)
    ax.legend(frameon=False, loc="lower left")
    save_figure(fig, out_path)


def plot_direction_outcome_rates(fraction_df: pd.DataFrame, out_path: Path) -> None:
    families = list(fraction_df["pair_family"].dropna().unique())
    fig, axes = plt.subplots(1, len(families), figsize=(13, 5.8), sharey=True, constrained_layout=True)
    if len(families) == 1:
        axes = [axes]

    for ax, pair_family in zip(axes, families):
        group_df = fraction_df[fraction_df["pair_family"] == pair_family].sort_values("target_fraction")
        color = DIRECTION_COLORS.get(pair_family, "#333333")
        ax.plot(
            group_df["target_fraction"],
            group_df["positive_rate"],
            marker="o",
            linewidth=2.4,
            markersize=7,
            linestyle="-",
            color=color,
            label="Positive rate",
        )
        ax.plot(
            group_df["target_fraction"],
            group_df["negative_rate"],
            marker="o",
            linewidth=2.4,
            markersize=7,
            linestyle="--",
            color=color,
            label="Negative rate",
        )
        ax.set_title(prettify_direction(pair_family))
        ax.set_xlabel("Target fraction")
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.25, linestyle="--")
        format_fraction_ticks(ax)
        ax.legend(frameon=False, loc="upper right")

    axes[0].set_ylabel("Rate")
    fig.suptitle("Positive vs Negative Transfer Rates by Direction")
    save_figure(fig, out_path)


def plot_direction_overview_bars(direction_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8), constrained_layout=True)

    ordered = direction_df.copy()
    ordered["direction_label"] = ordered["pair_family"].map(prettify_direction)

    axes[0].bar(
        ordered["direction_label"],
        ordered["mean_primary_gain"],
        color=[DIRECTION_COLORS.get(x, "#333333") for x in ordered["pair_family"]],
    )
    axes[0].axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    axes[0].set_title("Mean Transfer Gain")
    axes[0].set_ylabel("Mean gain (F1 attack)")
    axes[0].tick_params(axis="x", rotation=12)

    axes[1].bar(
        ordered["direction_label"],
        ordered["net_positive_minus_negative"],
        color=[DIRECTION_COLORS.get(x, "#333333") for x in ordered["pair_family"]],
    )
    axes[1].axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    axes[1].set_title("Positive Minus Negative Cases")
    axes[1].set_ylabel("Count difference")
    axes[1].tick_params(axis="x", rotation=12)

    for ax in axes:
        ax.grid(alpha=0.2, linestyle="--", axis="y")
        for container in ax.containers:
            ax.bar_label(container, fmt="%.4f" if ax is axes[0] else "%d", padding=3, fontsize=11)

    fig.suptitle("Direction-Level Transfer Overview", y=1.02)
    save_figure(fig, out_path)


def plot_case_gain_distribution(case_df: pd.DataFrame, out_path: Path) -> None:
    plot_df = case_df.copy()
    plot_df["direction_label"] = plot_df["pair_family"].map(prettify_direction)
    plot_df["transfer_label"] = plot_df["transfer_label"].str.title()

    fig, ax = plt.subplots(figsize=(13.5, 6.4), constrained_layout=True)
    sns.boxplot(
        data=plot_df,
        x="direction_label",
        y="primary_gain",
        hue="transfer_label",
        hue_order=TRANSFER_LABEL_ORDER,
        palette={
            "Positive": TRANSFER_LABEL_COLORS["positive"],
            "Neutral": TRANSFER_LABEL_COLORS["neutral"],
            "Negative": TRANSFER_LABEL_COLORS["negative"],
        },
        dodge=True,
        showcaps=True,
        fliersize=2.5,
        linewidth=1.0,
        width=0.72,
        ax=ax,
    )
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("")
    ax.set_ylabel("Transfer gain (F1 attack)")
    ax.set_title("Distribution of Pairwise Transfer Gain by Direction")
    ax.grid(alpha=0.2, linestyle="--", axis="y")
    ax.legend(frameon=False, title="", loc="upper right")
    save_figure(fig, out_path)


def main() -> None:
    args = parse_args()
    apply_plot_style()
    analysis_dir = Path(args.analysis_dir)
    out_dir = Path(args.out_dir) if args.out_dir else analysis_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    direction_df = pd.read_csv(analysis_dir / "direction_summary.csv")
    fraction_df = pd.read_csv(analysis_dir / "direction_fraction_summary.csv")
    case_df = pd.read_csv(analysis_dir / "combined_gain_table.csv")

    direction_df = add_direction_labels(direction_df)
    fraction_df = add_direction_labels(fraction_df)
    case_df = add_direction_labels(case_df)

    plot_direction_fraction_gain(
        fraction_df=fraction_df,
        out_path=out_dir / "direction_mean_gain_by_fraction.png",
    )
    plot_direction_outcome_rates(
        fraction_df=fraction_df,
        out_path=out_dir / "direction_positive_negative_rates.png",
    )
    plot_direction_overview_bars(
        direction_df=direction_df,
        out_path=out_dir / "direction_overview_bars.png",
    )
    plot_case_gain_distribution(
        case_df=case_df,
        out_path=out_dir / "direction_gain_distribution.png",
    )

    print(f"Saved combined direction figures to: {out_dir}")


if __name__ == "__main__":
    main()
