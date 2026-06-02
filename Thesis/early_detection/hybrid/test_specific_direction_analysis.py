from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
KNOWN_ABLATIONS = set(ABLATION_ORDER)
ABLATION_ALIASES = {
    "full_model": "full_model",
    "uniform_gating": "uniform_gating",
    "tabular_temporal": "tabular_temporal",
    "tabular_prototype": "tabular_prototype",
    "temporal_prototype": "temporal_prototype",
    "tabular_only": "tabular_only",
    "temporal_only": "temporal_only",
    "prototype_only": "prototype_only",
    "no_prototype": "tabular_temporal",
    "no_temporal": "tabular_prototype",
    "no_tabular": "temporal_prototype",
    "no_temporal__no_prototype": "tabular_only",
    "no_tabular__no_prototype": "temporal_only",
    "no_tabular__no_temporal": "prototype_only",
}
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Direction-specific subgroup analysis across ablations and seeds."
    )
    parser.add_argument(
        "--runs_dir",
        default="early_detection/hybrid/updated_arch2_more_tests",
        help="Directory containing ablation folders with seed_* runs.",
    )
    parser.add_argument(
        "--out_dir",
        default="early_detection/hybrid/updated_arch2_more_tests/test_specific_direction_analysis",
        help="Directory for analysis outputs.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["val", "test"],
        help="Which split to analyze.",
    )
    parser.add_argument(
        "--fractions",
        default="0.1,1.0",
        help="Comma-separated fractions to visualize, e.g. '0.1,0.2,1.0'.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=15,
        help="Maximum number of scenarios / attack categories to include per plot.",
    )
    return parser.parse_args()


def t_critical_95(n: int) -> float:
    if n <= 1:
        return 0.0
    df = n - 1
    if df in T_CRIT_95:
        return T_CRIT_95[df]
    return 1.96


def safe_to_numeric(series: pd.Series) -> pd.Series:
    try:
        return pd.to_numeric(series)
    except (TypeError, ValueError):
        return series


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
    for col in df.columns:
        if col in {"split", "direction", "seed", "ablation_name", "ablation_dir", "run_dir", "scenario", "attack_cat"}:
            continue
        df[col] = safe_to_numeric(df[col])
    return df


def normalize_ablation_name(value: object) -> str:
    text = str(value).strip()
    lowered = text.lower()
    if text in ABLATION_ALIASES:
        return ABLATION_ALIASES[text]
    if lowered in ABLATION_ALIASES:
        return ABLATION_ALIASES[lowered]
    if text in KNOWN_ABLATIONS:
        return text
    if lowered in KNOWN_ABLATIONS:
        return lowered
    return text


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
    out = pd.concat(rows, ignore_index=True)
    out["ablation_name"] = out["ablation_name"].map(normalize_ablation_name)
    return out


def summarize_metric(
    df: pd.DataFrame,
    group_cols: list[str],
    metric: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_key, group_df in df.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        values = pd.to_numeric(group_df[metric], errors="coerce").dropna()
        n = int(values.shape[0])
        if n == 0:
            continue
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if n > 1 else 0.0
        se = std / math.sqrt(n) if n > 1 else 0.0
        margin = t_critical_95(n) * se if n > 1 else 0.0
        rows.append(
            {
                **dict(zip(group_cols, group_key)),
                "metric": metric,
                "n": n,
                "mean": mean,
                "std": std,
                "ci_low": max(0.0, mean - margin),
                "ci_high": min(1.0, mean + margin),
                "ci_width": min(1.0, mean + margin) - max(0.0, mean - margin),
            }
        )
    return pd.DataFrame(rows)


def parse_fraction_list(text: str) -> list[float]:
    values = []
    for part in text.split(","):
        cleaned = part.strip()
        if not cleaned:
            continue
        values.append(float(cleaned))
    if not values:
        raise ValueError("No fractions were provided.")
    return values


def pick_top_groups(
    df: pd.DataFrame,
    subgroup_col: str,
    split: str,
    direction: str,
    top_k: int,
) -> list[str]:
    subset = df[(df["split"] == split) & (df["direction"] == direction)].copy()
    if subset.empty or "attack_support" not in subset.columns:
        return []
    full_fraction = pd.to_numeric(subset["fraction"], errors="coerce").max()
    subset = subset[pd.to_numeric(subset["fraction"], errors="coerce") == full_fraction]
    ordered = (
        subset.groupby(subgroup_col, as_index=False)["attack_support"]
        .mean()
        .sort_values("attack_support", ascending=False)
    )
    return ordered.head(top_k)[subgroup_col].tolist()


def plot_ablation_heatmap(
    summary_df: pd.DataFrame,
    subgroup_col: str,
    subgroup_values: list[str],
    direction: str,
    fraction: float,
    metric: str,
    out_path: Path,
    title: str,
) -> None:
    subset = summary_df[
        (summary_df["direction"] == direction)
        & (summary_df["fraction"] == fraction)
        & (summary_df["metric"] == metric)
        & (summary_df[subgroup_col].isin(subgroup_values))
    ].copy()
    if subset.empty:
        return
    subset["ablation_name"] = pd.Categorical(
        subset["ablation_name"], categories=ABLATION_ORDER, ordered=True
    )
    row_order = (
        subset.groupby(subgroup_col, as_index=False)["mean"]
        .max()
        .sort_values("mean", ascending=False)[subgroup_col]
        .tolist()
    )
    pivot = (
        subset.pivot_table(index=subgroup_col, columns="ablation_name", values="mean", aggfunc="first")
        .reindex(index=row_order, columns=ABLATION_ORDER)
    )

    fig_height = max(5, 0.45 * len(pivot.index))
    fig, ax = plt.subplots(figsize=(10.5, fig_height))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Ablation")
    ax.set_ylabel(subgroup_col)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=f"Mean {metric}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_best_ablation_heatmap(
    summary_df: pd.DataFrame,
    subgroup_col: str,
    subgroup_values: list[str],
    direction: str,
    fractions: list[float],
    metric: str,
    out_path: Path,
    title: str,
) -> None:
    subset = summary_df[
        (summary_df["direction"] == direction)
        & (summary_df["fraction"].isin(fractions))
        & (summary_df["metric"] == metric)
        & (summary_df[subgroup_col].isin(subgroup_values))
    ].copy()
    subset = subset[subset["ablation_name"].isin(ABLATION_ORDER)]
    if subset.empty:
        return

    rows = []
    for (subgroup, fraction), group_df in subset.groupby([subgroup_col, "fraction"], sort=False):
        winner = group_df.sort_values(["mean", "ci_width"], ascending=[False, True]).iloc[0]
        rows.append(
            {
                subgroup_col: subgroup,
                "fraction": fraction,
                "winner_index": ABLATION_ORDER.index(str(winner["ablation_name"])),
                "winner_name": str(winner["ablation_name"]),
                "winner_score": float(winner["mean"]),
            }
        )
    winner_df = pd.DataFrame(rows)
    if winner_df.empty:
        return
    row_order = (
        winner_df.groupby(subgroup_col, as_index=False)["winner_score"]
        .max()
        .sort_values("winner_score", ascending=False)[subgroup_col]
        .tolist()
    )
    pivot = (
        winner_df.pivot_table(index=subgroup_col, columns="fraction", values="winner_index", aggfunc="first")
        .reindex(index=row_order, columns=fractions)
    )
    cmap = plt.get_cmap("tab10", len(ABLATION_ORDER))
    fig_height = max(5, 0.45 * len(pivot.index))
    fig, ax = plt.subplots(figsize=(9, fig_height))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap=cmap, vmin=-0.5, vmax=len(ABLATION_ORDER) - 0.5)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(x) for x in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Fraction")
    ax.set_ylabel(subgroup_col)
    ax.set_title(title)
    for i, subgroup in enumerate(pivot.index):
        for j, fraction in enumerate(pivot.columns):
            winner_index = pivot.iloc[i, j]
            if pd.isna(winner_index):
                continue
            winner_name = ABLATION_ORDER[int(winner_index)]
            ax.text(j, i, winner_name.replace("_", "\n"), ha="center", va="center", fontsize=8, color="white")
    cbar = fig.colorbar(im, ax=ax, ticks=range(len(ABLATION_ORDER)))
    cbar.ax.set_yticklabels(ABLATION_ORDER)
    cbar.set_label("Best ablation")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_strength_tables(
    summary_df: pd.DataFrame,
    subgroup_col: str,
    direction: str,
    fractions: list[float],
    metric: str,
    out_dir: Path,
    prefix: str,
) -> None:
    subset = summary_df[
        (summary_df["direction"] == direction)
        & (summary_df["fraction"].isin(fractions))
        & (summary_df["metric"] == metric)
    ].copy()
    subset = subset[subset["ablation_name"].isin(ABLATION_ORDER)]
    if subset.empty:
        return

    ranking_rows = []
    winner_rows = []
    for (subgroup, fraction), group_df in subset.groupby([subgroup_col, "fraction"], sort=False):
        ranked = group_df.sort_values(["mean", "ci_width"], ascending=[False, True]).reset_index(drop=True)
        for rank_idx, (_, row) in enumerate(ranked.iterrows(), start=1):
            ranking_rows.append(
                {
                    subgroup_col: subgroup,
                    "fraction": fraction,
                    "rank": rank_idx,
                    "ablation_name": row["ablation_name"],
                    "mean": row["mean"],
                    "std": row["std"],
                    "ci_low": row["ci_low"],
                    "ci_high": row["ci_high"],
                    "ci_width": row["ci_width"],
                    "n": row["n"],
                }
            )
        winner = ranked.iloc[0]
        runner_up = ranked.iloc[1] if ranked.shape[0] > 1 else None
        winner_rows.append(
            {
                subgroup_col: subgroup,
                "fraction": fraction,
                "best_ablation": winner["ablation_name"],
                "best_mean": winner["mean"],
                "best_ci_width": winner["ci_width"],
                "runner_up_ablation": runner_up["ablation_name"] if runner_up is not None else "",
                "runner_up_mean": runner_up["mean"] if runner_up is not None else np.nan,
                "margin_vs_runner_up": winner["mean"] - runner_up["mean"] if runner_up is not None else np.nan,
            }
        )

    pd.DataFrame(ranking_rows).to_csv(out_dir / f"{prefix}_rankings.csv", index=False)
    pd.DataFrame(winner_rows).to_csv(out_dir / f"{prefix}_best_ablation_summary.csv", index=False)

    stability = (
        subset.groupby("ablation_name", as_index=False)
        .agg(
            mean_of_means=("mean", "mean"),
            mean_ci_width=("ci_width", "mean"),
            mean_std=("std", "mean"),
            n_groups=(subgroup_col, "nunique"),
        )
        .sort_values(["mean_of_means", "mean_ci_width"], ascending=[False, True])
    )
    stability.to_csv(out_dir / f"{prefix}_ablation_stability_summary.csv", index=False)


def analyze_direction(
    raw_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    subgroup_col: str,
    direction: str,
    split: str,
    fractions: list[float],
    top_k: int,
    out_dir: Path,
    title_prefix: str,
    prefix: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    top_groups = pick_top_groups(raw_df, subgroup_col, split, direction, top_k)
    if not top_groups:
        return

    filtered_summary = summary_df[
        (summary_df["split"] == split)
        & (summary_df["direction"] == direction)
        & (summary_df[subgroup_col].isin(top_groups))
    ].copy()
    filtered_summary.to_csv(out_dir / f"{prefix}_summary_long.csv", index=False)

    for fraction in fractions:
        fraction_subset = filtered_summary[filtered_summary["fraction"] == fraction]
        if fraction_subset.empty:
            continue
        plot_ablation_heatmap(
            fraction_subset,
            subgroup_col=subgroup_col,
            subgroup_values=top_groups,
            direction=direction,
            fraction=fraction,
            metric="f1_attack",
            out_path=out_dir / f"{prefix}_fraction_{str(fraction).replace('.', 'p')}_f1_heatmap.png",
            title=f"{title_prefix}: mean attack F1 by ablation ({split}, {direction}, fraction={fraction})",
        )

    plot_best_ablation_heatmap(
        filtered_summary,
        subgroup_col=subgroup_col,
        subgroup_values=top_groups,
        direction=direction,
        fractions=fractions,
        metric="f1_attack",
        out_path=out_dir / f"{prefix}_best_ablation_heatmap.png",
        title=f"{title_prefix}: best ablation by subgroup and fraction ({split}, {direction})",
    )

    make_strength_tables(
        filtered_summary,
        subgroup_col=subgroup_col,
        direction=direction,
        fractions=fractions,
        metric="f1_attack",
        out_dir=out_dir,
        prefix=prefix,
    )


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fractions = parse_fraction_list(args.fractions)

    iot23_raw = load_named_seed_summary(runs_dir, "overall_iot23_scenario_summary.csv")
    unsw_raw = load_named_seed_summary(runs_dir, "overall_unsw_attack_cat_summary.csv")

    iot23_summary = summarize_metric(
        iot23_raw,
        ["ablation_name", "direction", "split", "fraction", "scenario"],
        "f1_attack",
    )
    iot23_support = summarize_metric(
        iot23_raw,
        ["ablation_name", "direction", "split", "fraction", "scenario"],
        "attack_support",
    )
    iot23_summary = iot23_summary.merge(
        iot23_support[
            ["ablation_name", "direction", "split", "fraction", "scenario", "mean"]
        ].rename(columns={"mean": "attack_support"}),
        on=["ablation_name", "direction", "split", "fraction", "scenario"],
        how="left",
    )

    unsw_summary = summarize_metric(
        unsw_raw,
        ["ablation_name", "direction", "split", "fraction", "attack_cat"],
        "f1_attack",
    )
    unsw_support = summarize_metric(
        unsw_raw,
        ["ablation_name", "direction", "split", "fraction", "attack_cat"],
        "attack_support",
    )
    unsw_summary = unsw_summary.merge(
        unsw_support[
            ["ablation_name", "direction", "split", "fraction", "attack_cat", "mean"]
        ].rename(columns={"mean": "attack_support"}),
        on=["ablation_name", "direction", "split", "fraction", "attack_cat"],
        how="left",
    )

    iot23_summary.to_csv(out_dir / "iot23_direction_specific_summary.csv", index=False)
    unsw_summary.to_csv(out_dir / "unsw_direction_specific_summary.csv", index=False)

    analyze_direction(
        raw_df=iot23_raw,
        summary_df=iot23_summary,
        subgroup_col="scenario",
        direction="unsw_to_iot23",
        split=args.split,
        fractions=fractions,
        top_k=args.top_k,
        out_dir=out_dir / "unsw_to_iot23_iot23_scenarios",
        title_prefix="IoT-23 scenario strengths and weaknesses",
        prefix="unsw_to_iot23_iot23_scenarios",
    )
    analyze_direction(
        raw_df=unsw_raw,
        summary_df=unsw_summary,
        subgroup_col="attack_cat",
        direction="iot23_to_unsw",
        split=args.split,
        fractions=fractions,
        top_k=args.top_k,
        out_dir=out_dir / "iot23_to_unsw_attack_categories",
        title_prefix="UNSW attack-category strengths and weaknesses",
        prefix="iot23_to_unsw_attack_categories",
    )


if __name__ == "__main__":
    main()
