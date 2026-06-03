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


def summarize_absolute_performance(case_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    metric_specs = [
        ("source_only_f1_attack", "Source-only"),
        ("target_only_f1_attack", "Target-only"),
        ("transfer_f1_attack", "Transfer"),
    ]

    for metric_column, condition_label in metric_specs:
        subset = case_df[["pair_family", "target_fraction", metric_column]].copy()
        subset = subset.rename(columns={metric_column: "f1_attack"})
        subset["condition_label"] = condition_label
        rows.append(subset)

    long_df = pd.concat(rows, ignore_index=True)
    summary = (
        long_df.groupby(["pair_family", "target_fraction", "condition_label"], dropna=False)["f1_attack"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary["sem"] = summary["std"] / summary["count"].pow(0.5)
    summary["ci95"] = 1.96 * summary["sem"].fillna(0.0)
    return summary


def summarize_pair_fraction_gain(case_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        case_df.groupby(["pair_family", "pair_id", "target_fraction"], dropna=False)["primary_gain"]
        .mean()
        .reset_index(name="mean_primary_gain")
    )
    pair_order = (
        summary.groupby(["pair_family", "pair_id"], dropna=False)["mean_primary_gain"]
        .mean()
        .reset_index()
        .sort_values(["pair_family", "mean_primary_gain"], ascending=[True, False])
    )
    order_map = {
        pair_family: pair_rows["pair_id"].tolist()
        for pair_family, pair_rows in pair_order.groupby("pair_family", dropna=False)
    }
    return summary, order_map


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


def plot_hypothesis_absolute_performance(case_df: pd.DataFrame, out_path: Path) -> None:
    summary = summarize_absolute_performance(case_df)
    direction_order = [family for family in DIRECTION_LABELS if family in summary["pair_family"].unique()]
    condition_order = ["Target-only", "Transfer", "Source-only"]
    condition_styles = {
        "Target-only": {"color": "#333333", "linestyle": "-", "marker": "o"},
        "Transfer": {"color": "#d17a22", "linestyle": "-", "marker": "s"},
        "Source-only": {"color": "#7a7a7a", "linestyle": "--", "marker": "^"},
    }

    fig, axes = plt.subplots(
        1,
        len(direction_order),
        figsize=(13.5, 6.2),
        sharey=True,
        constrained_layout=True,
    )
    if len(direction_order) == 1:
        axes = [axes]

    for ax, pair_family in zip(axes, direction_order):
        panel_df = summary[summary["pair_family"] == pair_family].copy()
        for condition_label in condition_order:
            line_df = (
                panel_df[panel_df["condition_label"] == condition_label]
                .sort_values("target_fraction")
            )
            if line_df.empty:
                continue
            style = condition_styles[condition_label]
            ax.plot(
                line_df["target_fraction"],
                line_df["mean"],
                label=condition_label,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=2.3,
                markersize=6.5,
            )
            ax.fill_between(
                line_df["target_fraction"],
                line_df["mean"] - line_df["ci95"],
                line_df["mean"] + line_df["ci95"],
                color=style["color"],
                alpha=0.12,
                linewidth=0,
            )

        ax.set_title(prettify_direction(pair_family), fontsize=14)
        ax.set_xlabel("Target fraction")
        ax.grid(alpha=0.25, linestyle="--")
        format_fraction_ticks(ax)

    axes[0].set_ylabel("Mean attack-class F1")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            frameon=False,
            loc="outside lower center",
            ncol=3,
        )
    fig.suptitle(
        "Transfer Performance by Direction",
        y=1.03,
        fontsize=14,
    )
    save_figure(fig, out_path)


def shorten_pair_id(value: str, max_len: int = 52) -> str:
    text = str(value)
    replacements = {
        "iot23__source_group__train__all_train_mixed": "I23-ATM",
        "iot23__source_group__train__moderate_attack_mix": "I23-MAM",
        "iot23__source_group__train__benign_rich_mix": "I23-BRM",
        "unsw__service__test__missing": "U-S:miss",
        "unsw__service__test__http": "U-S:http",
        "unsw__service__test__dns": "U-S:dns",
        "unsw__proto__test__tcp": "U-P:tcp",
        "unsw__proto__test__udp": "U-P:udp",
        "unsw__service__train__missing": "Utr-S:miss",
        "unsw__service__train__http": "Utr-S:http",
        "unsw__service__train__dns": "Utr-S:dns",
        "unsw__service__train__ftp": "Utr-S:ftp",
        "unsw__service__train__ftp_data": "Utr-S:ftpd",
        "unsw__service__train__smtp": "Utr-S:smtp",
        "unsw__proto__train__tcp": "Utr-P:tcp",
        "unsw__proto__train__udp": "Utr-P:udp",
        "iot23__scenario__test__ctu_iot_malware_capture_34_1": "I23-T:34_1",
        "iot23__scenario__test__ctu_iot_malware_capture_43_1": "I23-T:43_1",
        "iot23__scenario__test__ctu_iot_malware_capture_48_1": "I23-T:48_1",
        "iot23__scenario__val__ctu_iot_malware_capture_49_1": "I23-V:49_1",
        "iot23__scenario__val__ctu_iot_malware_capture_52_1": "I23-V:52_1",
        "__TO__": " -> ",
        "TO": "->",
        "__": "|",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = text.replace("|", "")
    text = " ".join(text.split())
    if len(text) > max_len:
        text = text[: max_len - 3].rstrip() + "..."
    return text


def plot_pairwise_gain_heatmaps(case_df: pd.DataFrame, out_path: Path) -> None:
    summary, order_map = summarize_pair_fraction_gain(case_df)
    direction_order = [family for family in DIRECTION_LABELS if family in summary["pair_family"].unique()]
    fig, axes = plt.subplots(
        1,
        len(direction_order),
        figsize=(14.5, 7.8),
        constrained_layout=True,
    )
    if len(direction_order) == 1:
        axes = [axes]

    cmap = sns.diverging_palette(12, 133, s=95, l=45, as_cmap=True)
    vlim = max(abs(summary["mean_primary_gain"].min()), abs(summary["mean_primary_gain"].max()))
    vlim = max(vlim, 0.02)

    for ax, pair_family in zip(axes, direction_order):
        panel_df = summary[summary["pair_family"] == pair_family].copy()
        pair_ids = order_map.get(pair_family, [])
        pivot = (
            panel_df.pivot(index="pair_id", columns="target_fraction", values="mean_primary_gain")
            .reindex(index=pair_ids)
            .sort_index(axis=1)
        )
        yticklabels = [shorten_pair_id(idx) for idx in pivot.index]
        sns.heatmap(
            pivot,
            ax=ax,
            cmap=cmap,
            center=0.0,
            vmin=-vlim,
            vmax=vlim,
            linewidths=0.4,
            linecolor="#f0f0f0",
            cbar=ax is axes[-1],
            cbar_kws={"label": "Mean transfer gain (F1 attack)"} if ax is axes[-1] else None,
            xticklabels=[f"{float(col):.2f}" for col in pivot.columns],
            yticklabels=yticklabels,
        )
        ax.set_title(prettify_direction(pair_family), fontsize=14)
        ax.set_xlabel("Target fraction")
        ax.set_ylabel("Pair" if ax is axes[0] else "")
        ax.tick_params(axis="x", rotation=0)
        ax.tick_params(axis="y", labelsize=9)

    fig.suptitle("Pairwise Transfer Gain Heatmaps", y=1.02, fontsize=14)
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
    plot_hypothesis_absolute_performance(
        case_df=case_df,
        out_path=out_dir / "direction_hypothesis_absolute_performance.png",
    )
    plot_pairwise_gain_heatmaps(
        case_df=case_df,
        out_path=out_dir / "direction_pairwise_gain_heatmaps.png",
    )

    print(f"Saved combined direction figures to: {out_dir}")


if __name__ == "__main__":
    main()
