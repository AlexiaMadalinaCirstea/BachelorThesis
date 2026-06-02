from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DIRECTIONS = ["iot23_to_unsw", "unsw_to_iot23"]
MAIN_METRICS = ["f1_attack", "recall_attack", "precision_attack", "accuracy"]
GATE_WEIGHT_METRICS = ["mean_tabular_weight", "mean_temporal_weight", "mean_prototype_weight"]
BRANCH_WIN_METRICS = [
    "tabular_branch_win_rate",
    "temporal_branch_win_rate",
    "prototype_branch_win_rate",
]
BRANCH_LABELS = {
    "mean_tabular_weight": "tabular",
    "mean_temporal_weight": "temporal",
    "mean_prototype_weight": "prototype",
    "tabular_branch_win_rate": "tabular",
    "temporal_branch_win_rate": "temporal",
    "prototype_branch_win_rate": "prototype",
}
ABLATION_ORDER = [
    "full_model",
    "uniform_gating",
    "tabular_temporal",
    "tabular_prototype",
    "temporal_prototype",
    "tabular_only",
    "temporal_only",
    "prototype_only",
]
T_CRIT_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}
BOUNDED_UNIT_METRICS = {
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "precision_attack",
    "recall_attack",
    "f1_attack",
    "attack_rate",
    "mean_tabular_weight",
    "mean_temporal_weight",
    "mean_prototype_weight",
    "std_tabular_weight",
    "std_temporal_weight",
    "std_prototype_weight",
    "mean_tabular_confidence",
    "mean_temporal_confidence",
    "mean_prototype_confidence",
    "mean_branch_agreement",
    "mean_prototype_margin",
    "tabular_branch_win_rate",
    "temporal_branch_win_rate",
    "prototype_branch_win_rate",
    "first_true_positive_fraction",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="In-depth analyzer for hybrid multi-seed runs.")
    parser.add_argument(
        "--runs_dir",
        default="early_detection/hybrid/updated_arch2_more_tests",
        help="Directory containing ablation folders with seed_* runs.",
    )
    parser.add_argument(
        "--out_dir",
        default="early_detection/hybrid/updated_arch2_more_tests/hybrid_analyzer_in_depth",
        help="Directory for analyzer outputs.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["val", "test"],
        help="Default split to plot for the main figures.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=15,
        help="Maximum number of scenarios / attack categories to display in subgroup plots.",
    )
    return parser.parse_args()


def safe_to_numeric(series: pd.Series) -> pd.Series:
    try:
        return pd.to_numeric(series)
    except (TypeError, ValueError):
        return series


def t_critical_95(n: int) -> float:
    if n <= 1:
        return 0.0
    df = n - 1
    if df in T_CRIT_95:
        return T_CRIT_95[df]
    return 1.96


def clip_ci_bounds(metric: str, low: float, high: float) -> tuple[float, float]:
    if metric in BOUNDED_UNIT_METRICS:
        return max(0.0, low), min(1.0, high)
    return low, high


def canonical_seed_summary_paths(runs_dir: Path, filename: str) -> list[Path]:
    paths: list[Path] = []
    for path in runs_dir.rglob(filename):
        if path.parent.name.startswith("seed_"):
            paths.append(path)
    return sorted(paths)


def infer_seed_and_ablation(path: Path) -> tuple[str, str]:
    seed_name = path.parent.name
    ablation_name = path.parent.parent.name if path.parent.parent else "unknown_ablation"
    return seed_name, ablation_name


def read_seed_level_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    seed_name, ablation_name = infer_seed_and_ablation(path)
    df["seed"] = seed_name
    if "ablation_name" in df.columns:
        df["ablation_name"] = df["ablation_name"].fillna(ablation_name)
    else:
        df["ablation_name"] = ablation_name
    df["ablation_dir"] = ablation_name
    df["run_dir"] = str(path.parent)
    return df


def load_fraction_summaries(runs_dir: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in canonical_seed_summary_paths(runs_dir, "overall_fraction_summary.csv"):
        try:
            df = read_seed_level_summary(path)
        except Exception:
            continue
        if df.empty or "direction" not in df.columns:
            continue
        rows.append(df)
    if not rows:
        raise ValueError(f"No seed-level overall_fraction_summary.csv files found under {runs_dir}")

    summary_df = pd.concat(rows, ignore_index=True)
    for col in summary_df.columns:
        if col == "split" or col == "direction" or col == "seed" or col == "ablation_name" or col == "ablation_dir" or col == "run_dir":
            continue
        summary_df[col] = safe_to_numeric(summary_df[col])

    duplicate_cols = [
        col
        for col in ["ablation_name", "seed", "direction", "split", "fraction"]
        if col in summary_df.columns
    ]
    duplicate_mask = summary_df.duplicated(subset=duplicate_cols, keep=False)
    if duplicate_mask.any():
        preview = (
            summary_df.loc[duplicate_mask, duplicate_cols]
            .sort_values(duplicate_cols)
            .drop_duplicates()
        )
        raise ValueError(
            "Duplicate seed-level fraction summary rows detected. Duplicate keys include:\n"
            f"{preview.to_string(index=False)}"
        )
    return summary_df


def load_named_seed_summary(runs_dir: Path, filename: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in canonical_seed_summary_paths(runs_dir, filename):
        try:
            df = read_seed_level_summary(path)
        except Exception:
            continue
        if df.empty:
            continue
        rows.append(df)
    if not rows:
        raise ValueError(f"No seed-level {filename} files found under {runs_dir}")

    summary_df = pd.concat(rows, ignore_index=True)
    for col in summary_df.columns:
        if col in {"split", "direction", "seed", "ablation_name", "ablation_dir", "run_dir"}:
            continue
        summary_df[col] = safe_to_numeric(summary_df[col])
    return summary_df


def summarize_metrics(
    df: pd.DataFrame,
    group_cols: list[str],
    metrics: Iterable[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_key, group_df in df.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        base = dict(zip(group_cols, group_key))
        for metric in metrics:
            if metric not in group_df.columns:
                continue
            values = pd.to_numeric(group_df[metric], errors="coerce").dropna()
            n = int(values.shape[0])
            if n == 0:
                continue
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if n > 1 else 0.0
            se = std / math.sqrt(n) if n > 1 else 0.0
            margin = t_critical_95(n) * se if n > 1 else 0.0
            ci_low, ci_high = clip_ci_bounds(metric, mean - margin, mean + margin)
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "n": n,
                    "mean": mean,
                    "std": std,
                    "se": se,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )
    return pd.DataFrame(rows)


def sort_fraction_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "fraction" in out.columns:
        out["fraction"] = pd.to_numeric(out["fraction"], errors="coerce")
    return out.sort_values([col for col in ["direction", "ablation_name", "split", "fraction", "metric"] if col in out.columns])


def make_run_coverage(df: pd.DataFrame) -> pd.DataFrame:
    coverage = (
        df.groupby(["ablation_name", "direction"], as_index=False)["seed"]
        .agg(n_seeds="nunique", seeds=lambda s: ", ".join(sorted(set(s))))
    )
    coverage["ablation_name"] = pd.Categorical(coverage["ablation_name"], categories=ABLATION_ORDER, ordered=True)
    return coverage.sort_values(["ablation_name", "direction"]).reset_index(drop=True)


def metric_plot(
    summary_df: pd.DataFrame,
    metric: str,
    split: str,
    out_path: Path,
    band: str,
    ablations: Iterable[str] | None = None,
) -> None:
    subset = summary_df[(summary_df["metric"] == metric) & (summary_df["split"] == split)].copy()
    if ablations is not None:
        subset = subset[subset["ablation_name"].isin(list(ablations))]
    if subset.empty:
        return

    fig, axes = plt.subplots(1, len(DIRECTIONS), figsize=(16, 5), sharey=True)
    if len(DIRECTIONS) == 1:
        axes = [axes]
    for ax, direction in zip(axes, DIRECTIONS):
        direction_df = subset[subset["direction"] == direction].copy()
        if direction_df.empty:
            ax.set_visible(False)
            continue
        direction_df["ablation_name"] = pd.Categorical(
            direction_df["ablation_name"], categories=ABLATION_ORDER, ordered=True
        )
        for _, group in direction_df.sort_values(["ablation_name", "fraction"]).groupby("ablation_name", sort=False):
            label = str(group["ablation_name"].iloc[0])
            ax.plot(group["fraction"], group["mean"], marker="o", label=label)
            if band == "std":
                low = group["mean"] - group["std"]
                high = group["mean"] + group["std"]
            else:
                low = group["ci_low"]
                high = group["ci_high"]
            ax.fill_between(group["fraction"], low, high, alpha=0.15)
        ax.set_title(direction)
        ax.set_xlabel("Fraction")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(metric)
    fig.suptitle(f"{metric} across fractions ({split}, mean +/- {band})")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout(rect=[0, 0, 0.88, 0.95])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def full_vs_uniform_plot(summary_df: pd.DataFrame, split: str, out_dir: Path) -> None:
    focus = ["full_model", "uniform_gating"]
    for metric in MAIN_METRICS:
        out_path = out_dir / f"{metric}_{split}_full_vs_uniform_ci.png"
        metric_plot(summary_df, metric, split, out_path, band="ci", ablations=focus)


def gate_behavior_plot(summary_df: pd.DataFrame, split: str, out_path: Path) -> None:
    subset = summary_df[
        (summary_df["split"] == split)
        & (summary_df["ablation_name"] == "full_model")
        & (summary_df["metric"].isin(GATE_WEIGHT_METRICS))
    ].copy()
    if subset.empty:
        return

    fig, axes = plt.subplots(1, len(DIRECTIONS), figsize=(14, 4.8), sharey=True)
    for ax, direction in zip(axes, DIRECTIONS):
        direction_df = subset[subset["direction"] == direction].copy()
        if direction_df.empty:
            ax.set_visible(False)
            continue
        for metric in GATE_WEIGHT_METRICS:
            group = direction_df[direction_df["metric"] == metric].sort_values("fraction")
            if group.empty:
                continue
            label = BRANCH_LABELS[metric]
            ax.plot(group["fraction"], group["mean"], marker="o", label=label)
            ax.fill_between(group["fraction"], group["ci_low"], group["ci_high"], alpha=0.15)
        ax.set_title(direction)
        ax.set_xlabel("Fraction")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Mean gate weight")
    fig.suptitle(f"Full-model gate behavior by fraction ({split})")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout(rect=[0, 0, 0.88, 0.95])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def branch_dominance_plot(summary_df: pd.DataFrame, split: str, out_path: Path) -> None:
    subset = summary_df[
        (summary_df["split"] == split)
        & (summary_df["ablation_name"] == "full_model")
        & (summary_df["metric"].isin(BRANCH_WIN_METRICS))
    ].copy()
    if subset.empty:
        return

    fig, axes = plt.subplots(1, len(DIRECTIONS), figsize=(14, 4.8), sharey=True)
    for ax, direction in zip(axes, DIRECTIONS):
        direction_df = subset[subset["direction"] == direction].copy()
        if direction_df.empty:
            ax.set_visible(False)
            continue
        for metric in BRANCH_WIN_METRICS:
            group = direction_df[direction_df["metric"] == metric].sort_values("fraction")
            if group.empty:
                continue
            label = BRANCH_LABELS[metric]
            ax.plot(group["fraction"], group["mean"], marker="o", label=label)
            ax.fill_between(group["fraction"], group["ci_low"], group["ci_high"], alpha=0.15)
        ax.set_title(direction)
        ax.set_xlabel("Fraction")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Branch win rate")
    fig.suptitle(f"Full-model branch dominance by fraction ({split})")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout(rect=[0, 0, 0.88, 0.95])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def top_groups_by_support(
    summary_df: pd.DataFrame,
    subgroup_col: str,
    split: str,
    direction: str,
    top_k: int,
    ablation_name: str | None = None,
) -> list[str]:
    subset = summary_df[
        (summary_df["split"] == split)
        & (summary_df["direction"] == direction)
        & (summary_df["metric"] == "attack_support")
    ].copy()
    if ablation_name is not None:
        subset = subset[subset["ablation_name"] == ablation_name]
    if subset.empty:
        return []
    full_fraction = subset["fraction"].max()
    subset = subset[subset["fraction"] == full_fraction]
    ordered = (
        subset.groupby(subgroup_col, as_index=False)["mean"]
        .mean()
        .sort_values("mean", ascending=False)
    )
    return ordered.head(top_k)[subgroup_col].tolist()


def subgroup_heatmap_plot(
    summary_df: pd.DataFrame,
    subgroup_col: str,
    split: str,
    direction: str,
    top_k: int,
    out_path: Path,
    title_prefix: str,
    ablation_name: str | None = None,
) -> None:
    focus_groups = top_groups_by_support(
        summary_df, subgroup_col, split, direction, top_k, ablation_name=ablation_name
    )
    if not focus_groups:
        return
    subset = summary_df[
        (summary_df["split"] == split)
        & (summary_df["direction"] == direction)
        & (summary_df["metric"] == "f1_attack")
        & (summary_df[subgroup_col].isin(focus_groups))
    ].copy()
    if ablation_name is not None:
        subset = subset[subset["ablation_name"] == ablation_name]
    if subset.empty:
        return

    fraction_order = sorted(pd.to_numeric(subset["fraction"], errors="coerce").dropna().unique().tolist())
    group_order = (
        subset[subset["fraction"] == max(fraction_order)]
        .sort_values("mean", ascending=False)[subgroup_col]
        .tolist()
    )
    pivot = (
        subset.pivot_table(index=subgroup_col, columns="fraction", values="mean", aggfunc="mean")
        .reindex(index=group_order, columns=fraction_order)
    )
    fig_height = max(5, 0.45 * len(pivot.index))
    fig, ax = plt.subplots(figsize=(10, fig_height))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(x) for x in pivot.columns], rotation=0)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Fraction")
    ax.set_ylabel(subgroup_col)
    title = f"{title_prefix}: mean attack F1 ({split}, {direction})"
    if ablation_name is not None:
        title += f"\n{ablation_name}"
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Mean attack F1")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def subgroup_full_fraction_bar_plot(
    summary_df: pd.DataFrame,
    subgroup_col: str,
    split: str,
    direction: str,
    top_k: int,
    out_path: Path,
    title_prefix: str,
    ablation_name: str | None = None,
) -> None:
    focus_groups = top_groups_by_support(
        summary_df, subgroup_col, split, direction, top_k, ablation_name=ablation_name
    )
    if not focus_groups:
        return
    subset = summary_df[
        (summary_df["split"] == split)
        & (summary_df["direction"] == direction)
        & (summary_df["metric"] == "f1_attack")
        & (summary_df[subgroup_col].isin(focus_groups))
    ].copy()
    if ablation_name is not None:
        subset = subset[subset["ablation_name"] == ablation_name]
    if subset.empty:
        return
    full_fraction = subset["fraction"].max()
    subset = subset[subset["fraction"] == full_fraction].sort_values("mean", ascending=False)

    fig_height = max(5, 0.45 * len(subset))
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ypos = np.arange(len(subset))
    ax.barh(ypos, subset["mean"], xerr=[subset["mean"] - subset["ci_low"], subset["ci_high"] - subset["mean"]], alpha=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels(subset[subgroup_col])
    ax.invert_yaxis()
    ax.set_xlabel("Mean attack F1")
    title = f"{title_prefix}: full-fraction attack F1 ({split}, {direction})"
    if ablation_name is not None:
        title += f"\n{ablation_name}"
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_subgroup_plots_per_ablation(
    summary_df: pd.DataFrame,
    subgroup_col: str,
    split: str,
    direction: str,
    top_k: int,
    out_dir: Path,
    title_prefix: str,
    filename_prefix: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    present_ablations = [
        name for name in ABLATION_ORDER if name in set(summary_df["ablation_name"].astype(str))
    ]
    for ablation_name in present_ablations:
        subgroup_heatmap_plot(
            summary_df,
            subgroup_col=subgroup_col,
            split=split,
            direction=direction,
            top_k=top_k,
            out_path=out_dir / f"{filename_prefix}_{ablation_name}_heatmap.png",
            title_prefix=title_prefix,
            ablation_name=ablation_name,
        )
        subgroup_full_fraction_bar_plot(
            summary_df,
            subgroup_col=subgroup_col,
            split=split,
            direction=direction,
            top_k=top_k,
            out_path=out_dir / f"{filename_prefix}_{ablation_name}_full_fraction.png",
            title_prefix=title_prefix,
            ablation_name=ablation_name,
        )


def write_summary_table(summary_df: pd.DataFrame, out_path: Path, split: str) -> None:
    subset = summary_df[
        (summary_df["split"] == split)
        & (summary_df["fraction"] == 1.0)
        & (summary_df["metric"].isin(MAIN_METRICS))
    ].copy()
    if subset.empty:
        return
    wide = subset.pivot_table(
        index=["direction", "ablation_name"],
        columns="metric",
        values=["mean", "std", "ci_low", "ci_high", "n"],
        aggfunc="first",
    )
    wide.sort_index().to_csv(out_path)


def save_long_summary(df: pd.DataFrame, out_path: Path) -> None:
    out = df.copy()
    for col in ["ablation_name", "direction", "split", "metric"]:
        if col in out.columns:
            out[col] = out[col].astype(str)
    out.sort_values([col for col in ["ablation_name", "direction", "split", "metric", "fraction"] if col in out.columns]).to_csv(out_path, index=False)


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fraction_df = sort_fraction_frame(load_fraction_summaries(runs_dir))
    fraction_df.to_csv(out_dir / "seed_level_fraction_summaries.csv", index=False)

    coverage_df = make_run_coverage(fraction_df)
    coverage_df.to_csv(out_dir / "run_coverage.csv", index=False)

    fraction_metrics = MAIN_METRICS + GATE_WEIGHT_METRICS + BRANCH_WIN_METRICS + [
        "mean_branch_agreement",
        "mean_prototype_margin",
        "first_true_positive_fraction",
        "attack_support",
    ]
    aggregated_fraction = summarize_metrics(
        fraction_df,
        ["ablation_name", "direction", "split", "fraction"],
        fraction_metrics,
    )
    save_long_summary(aggregated_fraction, out_dir / "fraction_metric_summary_long.csv")
    write_summary_table(aggregated_fraction, out_dir / "fraction_metric_summary_full_fraction.csv", args.split)

    for metric in MAIN_METRICS:
        metric_plot(
            aggregated_fraction,
            metric,
            args.split,
            plots_dir / f"{metric}_{args.split}_mean_std.png",
            band="std",
        )
        metric_plot(
            aggregated_fraction,
            metric,
            args.split,
            plots_dir / f"{metric}_{args.split}_mean_ci.png",
            band="ci",
        )

    full_vs_uniform_plot(aggregated_fraction, args.split, plots_dir)
    gate_behavior_plot(aggregated_fraction, args.split, plots_dir / f"gate_behavior_{args.split}_full_model.png")
    branch_dominance_plot(aggregated_fraction, args.split, plots_dir / f"branch_dominance_{args.split}_full_model.png")

    iot23_scenario_df = load_named_seed_summary(runs_dir, "overall_iot23_scenario_summary.csv")
    iot23_scenario_df.to_csv(out_dir / "seed_level_iot23_scenario_summaries.csv", index=False)
    iot23_scenario_summary = summarize_metrics(
        iot23_scenario_df,
        ["ablation_name", "direction", "split", "fraction", "scenario"],
        ["f1_attack", "recall_attack", "precision_attack", "accuracy", "first_true_positive_fraction", "attack_support"],
    )
    save_long_summary(iot23_scenario_summary, out_dir / "iot23_scenario_metric_summary_long.csv")
    make_subgroup_plots_per_ablation(
        iot23_scenario_summary,
        subgroup_col="scenario",
        split=args.split,
        direction="unsw_to_iot23",
        top_k=args.top_k,
        out_dir=plots_dir / f"iot23_scenarios_{args.split}_unsw_to_iot23",
        title_prefix="IoT-23 scenario transfer",
        filename_prefix=f"iot23_scenarios_{args.split}_unsw_to_iot23",
    )

    unsw_attack_df = load_named_seed_summary(runs_dir, "overall_unsw_attack_cat_summary.csv")
    unsw_attack_df.to_csv(out_dir / "seed_level_unsw_attack_cat_summaries.csv", index=False)
    unsw_attack_summary = summarize_metrics(
        unsw_attack_df,
        ["ablation_name", "direction", "split", "fraction", "attack_cat"],
        ["f1_attack", "recall_attack", "precision_attack", "accuracy", "first_true_positive_fraction", "attack_support"],
    )
    save_long_summary(unsw_attack_summary, out_dir / "unsw_attack_cat_metric_summary_long.csv")
    make_subgroup_plots_per_ablation(
        unsw_attack_summary,
        subgroup_col="attack_cat",
        split=args.split,
        direction="iot23_to_unsw",
        top_k=args.top_k,
        out_dir=plots_dir / f"unsw_attack_categories_{args.split}_iot23_to_unsw",
        title_prefix="UNSW attack-category transfer",
        filename_prefix=f"unsw_attack_categories_{args.split}_iot23_to_unsw",
    )


if __name__ == "__main__":
    main()
