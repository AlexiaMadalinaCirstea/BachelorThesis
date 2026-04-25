from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import pandas as pd


METRIC_COLUMNS = [
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "precision_attack",
    "recall_attack",
    "f1_attack",
]

PAIR_LEVEL_KEYS = [
    "pair_id",
    "pair_family",
    "source_domain_id",
    "target_test_domain_id",
    "resolved_source_train_ids",
    "resolved_target_train_ids",
    "target_train_resolution",
    "seed",
    "source_attack_rate",
    "target_attack_rate",
    "attack_rate_gap",
    "abs_attack_rate_gap",
]

RUN_LEVEL_KEYS = PAIR_LEVEL_KEYS + [
    "target_fraction",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze pairwise transfer-learning hypothesis runs and export gain/compatibility tables."
        )
    )
    parser.add_argument(
        "--run_dir",
        default="transfer_learning/hypothesis/pairwise_runs_iot23_to_unsw_seed42",
        help="Completed pairwise run directory containing pairwise_hypothesis_summary.csv.",
    )
    parser.add_argument(
        "--domain_pairs",
        default="transfer_learning/hypothesis/generated_pairs_heavier/domain_pairs.csv",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Optional output directory. Defaults to <run_dir>/analysis.",
    )
    parser.add_argument(
        "--gain_metric",
        default="f1_attack",
        choices=METRIC_COLUMNS,
        help="Primary metric used for positive/negative transfer summaries.",
    )
    parser.add_argument(
        "--top_k",
        nargs="+",
        type=int,
        default=[3, 5],
        help="Top-k values used for feature-overlap compatibility metrics.",
    )
    parser.add_argument(
        "--benefit_margin",
        type=float,
        default=0.0,
        help="Minimum gain required to count as positive transfer.",
    )
    return parser.parse_args()


def slugify(value: object, max_len: int = 120) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if len(text) > max_len:
        text = text[:max_len].rstrip("_")
    return text or "item"


def safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_inputs(run_dir: Path, domain_pairs_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_path = run_dir / "pairwise_hypothesis_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary CSV: {summary_path}")

    summary_df = pd.read_csv(summary_path)
    pairs_df = pd.read_csv(domain_pairs_path)

    required_summary = {"pair_id", "condition", "seed", "target_fraction"} | set(METRIC_COLUMNS)
    missing_summary = required_summary - set(summary_df.columns)
    if missing_summary:
        raise ValueError(f"Summary CSV is missing columns: {sorted(missing_summary)}")

    required_pairs = {
        "pair_id",
        "pair_family",
        "source_domain_id",
        "target_domain_id",
        "source_attack_rate",
        "target_attack_rate",
    }
    missing_pairs = required_pairs - set(pairs_df.columns)
    if missing_pairs:
        raise ValueError(f"Domain pairs CSV is missing columns: {sorted(missing_pairs)}")

    pairs_df = pairs_df.rename(columns={"target_domain_id": "target_test_domain_id"}).copy()
    pairs_df["attack_rate_gap"] = pairs_df["source_attack_rate"] - pairs_df["target_attack_rate"]
    pairs_df["abs_attack_rate_gap"] = pairs_df["attack_rate_gap"].abs()
    return summary_df, pairs_df


def prepare_valid_summary(summary_df: pd.DataFrame, pairs_df: pd.DataFrame) -> pd.DataFrame:
    merged = summary_df.merge(
        pairs_df[
            [
                "pair_id",
                "pair_family",
                "source_domain_id",
                "target_test_domain_id",
                "source_attack_rate",
                "target_attack_rate",
                "attack_rate_gap",
                "abs_attack_rate_gap",
            ]
        ],
        on=["pair_id", "pair_family", "source_domain_id", "target_test_domain_id"],
        how="left",
    )

    allowed_conditions = {
        "source_only",
        "target_only_updated",
        "transfer_learning_updated",
    }
    merged = merged[merged["condition"].isin(allowed_conditions)].copy()
    merged["is_valid_metric_row"] = merged[METRIC_COLUMNS].notna().all(axis=1)
    merged = merged[merged["is_valid_metric_row"]].copy()
    return merged


def rename_with_prefix(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    rename_map = {}
    for column in METRIC_COLUMNS + [
        "direction",
        "n_source_train",
        "n_target_train",
        "n_target_test",
        "n_features",
        "n_target_fit",
        "n_target_calibration",
        "selected_threshold",
        "threshold_selection_mode",
    ]:
        if column in df.columns:
            rename_map[column] = f"{prefix}_{column}"
    return df.rename(columns=rename_map)


def build_gain_table(valid_df: pd.DataFrame) -> pd.DataFrame:
    source_base = valid_df[valid_df["condition"] == "source_only"].copy()
    target_base = valid_df[valid_df["condition"] == "target_only_updated"].copy()
    transfer_base = valid_df[valid_df["condition"] == "transfer_learning_updated"].copy()

    source_keep = [
        column
        for column in PAIR_LEVEL_KEYS
        if column in source_base.columns
    ] + [column for column in METRIC_COLUMNS + ["direction"] if column in source_base.columns]
    target_keep = [
        column
        for column in RUN_LEVEL_KEYS
        if column in target_base.columns
    ] + [
        column
        for column in METRIC_COLUMNS
        + ["direction", "n_target_fit", "n_target_calibration", "selected_threshold", "threshold_selection_mode"]
        if column in target_base.columns
    ]
    transfer_keep = [
        column
        for column in RUN_LEVEL_KEYS
        if column in transfer_base.columns
    ] + [
        column
        for column in METRIC_COLUMNS
        + ["direction", "n_source_train", "n_target_fit", "n_target_calibration", "selected_threshold", "threshold_selection_mode"]
        if column in transfer_base.columns
    ]

    source_only = rename_with_prefix(source_base[source_keep].drop_duplicates(), "source_only")
    target_only = rename_with_prefix(target_base[target_keep].drop_duplicates(), "target_only")
    transfer = rename_with_prefix(transfer_base[transfer_keep].drop_duplicates(), "transfer")

    merged = target_only.merge(
        transfer,
        on=RUN_LEVEL_KEYS,
        how="inner",
        suffixes=("", "_dup"),
    )
    merged = merged.merge(
        source_only,
        on=PAIR_LEVEL_KEYS,
        how="left",
    )

    for metric in METRIC_COLUMNS:
        merged[f"gain_{metric}"] = merged[f"transfer_{metric}"] - merged[f"target_only_{metric}"]
        merged[f"delta_vs_source_{metric}"] = merged[f"transfer_{metric}"] - merged[f"source_only_{metric}"]
        merged[f"target_minus_source_{metric}"] = merged[f"target_only_{metric}"] - merged[f"source_only_{metric}"]

    return merged.sort_values(["target_fraction", "pair_id", "seed"]).reset_index(drop=True)


def compute_rank_spearman(values_a: pd.Series, values_b: pd.Series) -> float | None:
    aligned = pd.concat([values_a, values_b], axis=1, join="inner").fillna(0.0)
    if aligned.empty:
        return None
    rank_a = aligned.iloc[:, 0].rank(method="average", ascending=False)
    rank_b = aligned.iloc[:, 1].rank(method="average", ascending=False)
    corr = rank_a.corr(rank_b, method="pearson")
    if pd.isna(corr):
        return None
    return float(corr)


def compute_top_k_metrics(series_a: pd.Series, series_b: pd.Series, k: int) -> dict[str, float | int | None]:
    top_a = list(series_a.sort_values(ascending=False).head(k).index)
    top_b = list(series_b.sort_values(ascending=False).head(k).index)
    set_a = set(top_a)
    set_b = set(top_b)
    overlap = len(set_a & set_b)
    union = len(set_a | set_b)
    return {
        f"top_{k}_overlap_count": overlap,
        f"top_{k}_overlap_ratio": overlap / float(k) if k > 0 else None,
        f"top_{k}_jaccard": overlap / float(union) if union > 0 else None,
    }


def load_feature_series(path: Path) -> pd.Series | None:
    df = safe_read_csv(path)
    if df is None or df.empty or "feature" not in df.columns or "importance" not in df.columns:
        return None
    series = df.groupby("feature", dropna=False)["importance"].sum()
    return series.astype(float).sort_values(ascending=False)


def resolve_pair_dir_map(run_dir: Path, pair_ids: list[str]) -> dict[str, Path]:
    slug_to_pair_id = {slugify(pair_id): pair_id for pair_id in pair_ids}
    pair_map: dict[str, Path] = {}
    for pair_dir in sorted(run_dir.glob("pair_*")):
        if not pair_dir.is_dir():
            continue
        match = re.match(r"pair_\d+__(.+)$", pair_dir.name)
        if not match:
            continue
        pair_slug = match.group(1)
        pair_id = slug_to_pair_id.get(pair_slug)
        if pair_id is not None:
            pair_map[pair_id] = pair_dir
    return pair_map


def fraction_slug(value: float) -> str:
    return str(value).replace(".", "p")


def compute_compatibility_table(gain_df: pd.DataFrame, run_dir: Path, top_k_values: list[int]) -> pd.DataFrame:
    pair_dir_map = resolve_pair_dir_map(run_dir, gain_df["pair_id"].dropna().unique().tolist())
    records: list[dict[str, object]] = []

    for row in gain_df.itertuples(index=False):
        record = {
            "pair_id": row.pair_id,
            "seed": row.seed,
            "target_fraction": row.target_fraction,
        }
        pair_dir = pair_dir_map.get(row.pair_id)
        if pair_dir is None:
            records.append(record)
            continue

        seed_dir = pair_dir / f"seed_{int(row.seed)}"
        frac_slug = fraction_slug(row.target_fraction)

        target_path = seed_dir / f"target_only_updated_frac_{frac_slug}" / "feature_importance.csv"
        transfer_pretrain_path = seed_dir / f"transfer_learning_updated_frac_{frac_slug}" / "feature_importance_pretrain.csv"
        transfer_adapted_path = seed_dir / f"transfer_learning_updated_frac_{frac_slug}" / "feature_importance.csv"
        source_only_path = seed_dir / "source_only" / "feature_importance.csv"

        target_series = load_feature_series(target_path)
        pretrain_series = load_feature_series(transfer_pretrain_path)
        adapted_series = load_feature_series(transfer_adapted_path)
        source_series = load_feature_series(source_only_path)

        if target_series is not None and pretrain_series is not None:
            aligned = pd.concat([pretrain_series, target_series], axis=1).fillna(0.0)
            aligned.columns = ["pretrain", "target"]
            record["pretrain_target_rank_spearman"] = compute_rank_spearman(
                aligned["pretrain"], aligned["target"]
            )
            for k in top_k_values:
                metrics = compute_top_k_metrics(aligned["pretrain"], aligned["target"], k)
                for key, value in metrics.items():
                    record[f"pretrain_target_{key}"] = value

        if target_series is not None and adapted_series is not None:
            aligned = pd.concat([adapted_series, target_series], axis=1).fillna(0.0)
            aligned.columns = ["adapted", "target"]
            record["adapted_target_rank_spearman"] = compute_rank_spearman(
                aligned["adapted"], aligned["target"]
            )

        if target_series is not None and source_series is not None:
            aligned = pd.concat([source_series, target_series], axis=1).fillna(0.0)
            aligned.columns = ["source", "target"]
            record["source_target_rank_spearman"] = compute_rank_spearman(
                aligned["source"], aligned["target"]
            )

        records.append(record)

    return pd.DataFrame(records)


def add_transfer_labels(gain_df: pd.DataFrame, gain_metric: str, benefit_margin: float) -> pd.DataFrame:
    label_col = f"gain_{gain_metric}"
    gain_df = gain_df.copy()
    gain_df["primary_gain"] = gain_df[label_col]
    gain_df["positive_transfer"] = gain_df["primary_gain"] > benefit_margin
    gain_df["negative_transfer"] = gain_df["primary_gain"] < -benefit_margin
    gain_df["neutral_transfer"] = ~(gain_df["positive_transfer"] | gain_df["negative_transfer"])
    gain_df["transfer_label"] = "neutral"
    gain_df.loc[gain_df["positive_transfer"], "transfer_label"] = "positive"
    gain_df.loc[gain_df["negative_transfer"], "transfer_label"] = "negative"
    return gain_df


def summarize_by_fraction(gain_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        gain_df.groupby("target_fraction", dropna=False)
        .agg(
            n_runs=("pair_id", "count"),
            n_pairs=("pair_id", "nunique"),
            positive_transfer=("positive_transfer", "sum"),
            negative_transfer=("negative_transfer", "sum"),
            neutral_transfer=("neutral_transfer", "sum"),
            mean_primary_gain=("primary_gain", "mean"),
            median_primary_gain=("primary_gain", "median"),
            mean_source_only_f1_attack=("source_only_f1_attack", "mean"),
            mean_target_only_f1_attack=("target_only_f1_attack", "mean"),
            mean_transfer_f1_attack=("transfer_f1_attack", "mean"),
            mean_abs_attack_rate_gap=("abs_attack_rate_gap", "mean"),
            mean_pretrain_target_rank_spearman=("pretrain_target_rank_spearman", "mean"),
        )
        .reset_index()
    )
    summary["positive_rate"] = summary["positive_transfer"] / summary["n_runs"]
    summary["negative_rate"] = summary["negative_transfer"] / summary["n_runs"]
    return summary.sort_values("target_fraction").reset_index(drop=True)


def summarize_by_pair(gain_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        gain_df.groupby(["pair_id", "pair_family", "source_domain_id", "target_test_domain_id"], dropna=False)
        .agg(
            n_runs=("target_fraction", "count"),
            fractions_seen=("target_fraction", lambda values: "|".join(str(v) for v in sorted(set(values)))),
            mean_primary_gain=("primary_gain", "mean"),
            median_primary_gain=("primary_gain", "median"),
            max_primary_gain=("primary_gain", "max"),
            min_primary_gain=("primary_gain", "min"),
            positive_transfer=("positive_transfer", "sum"),
            negative_transfer=("negative_transfer", "sum"),
            mean_source_only_f1_attack=("source_only_f1_attack", "mean"),
            mean_target_only_f1_attack=("target_only_f1_attack", "mean"),
            mean_transfer_f1_attack=("transfer_f1_attack", "mean"),
            mean_pretrain_target_rank_spearman=("pretrain_target_rank_spearman", "mean"),
            abs_attack_rate_gap=("abs_attack_rate_gap", "mean"),
        )
        .reset_index()
        .sort_values(["mean_primary_gain", "max_primary_gain"], ascending=[False, False])
    )
    return summary.reset_index(drop=True)


def summarize_metric_gains(gain_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric in METRIC_COLUMNS:
        gain_col = f"gain_{metric}"
        for target_fraction, group_df in gain_df.groupby("target_fraction", dropna=False):
            rows.append(
                {
                    "metric": metric,
                    "target_fraction": target_fraction,
                    "n_runs": len(group_df),
                    "mean_gain": group_df[gain_col].mean(),
                    "median_gain": group_df[gain_col].median(),
                    "max_gain": group_df[gain_col].max(),
                    "min_gain": group_df[gain_col].min(),
                }
            )
    return pd.DataFrame(rows).sort_values(["metric", "target_fraction"]).reset_index(drop=True)


def build_predictor_correlation_table(gain_df: pd.DataFrame) -> pd.DataFrame:
    predictors = [
        "source_only_accuracy",
        "source_only_f1_macro",
        "source_only_f1_attack",
        "target_minus_source_f1_attack",
        "abs_attack_rate_gap",
        "pretrain_target_rank_spearman",
        "adapted_target_rank_spearman",
        "source_target_rank_spearman",
        "pretrain_target_top_3_overlap_ratio",
        "pretrain_target_top_5_overlap_ratio",
    ]

    scopes: list[tuple[str, pd.DataFrame]] = [("overall", gain_df)]
    scopes.extend(
        (f"target_fraction={target_fraction}", group_df.copy())
        for target_fraction, group_df in gain_df.groupby("target_fraction", dropna=False)
    )

    rows: list[dict[str, object]] = []
    for scope_name, scope_df in scopes:
        for predictor in predictors:
            if predictor not in scope_df.columns:
                continue

            subset = scope_df[[predictor, "primary_gain"]].dropna()
            if len(subset) < 3:
                continue

            pearson = subset[predictor].corr(subset["primary_gain"], method="pearson")
            spearman = subset[predictor].corr(subset["primary_gain"], method="spearman")
            rows.append(
                {
                    "scope": scope_name,
                    "predictor": predictor,
                    "n_runs": len(subset),
                    "pearson_corr": pearson,
                    "spearman_corr": spearman,
                    "abs_pearson_corr": abs(pearson) if pd.notna(pearson) else None,
                    "abs_spearman_corr": abs(spearman) if pd.notna(spearman) else None,
                }
            )

    corr_df = pd.DataFrame(rows)
    if corr_df.empty:
        return corr_df
    return corr_df.sort_values(["scope", "abs_spearman_corr"], ascending=[True, False]).reset_index(drop=True)


def format_case_line(row: pd.Series, metric_name: str) -> str:
    return (
        f"{row['pair_id']} | seed={int(row['seed'])} | frac={row['target_fraction']:.2f} | "
        f"gain_{metric_name}={row['primary_gain']:.4f} | "
        f"target={row[f'target_only_{metric_name}']:.4f} | "
        f"transfer={row[f'transfer_{metric_name}']:.4f}"
    )


def write_text_summary(
    out_path: Path,
    gain_df: pd.DataFrame,
    fraction_summary: pd.DataFrame,
    predictor_df: pd.DataFrame,
    gain_metric: str,
) -> None:
    top_positive = gain_df.sort_values("primary_gain", ascending=False).head(5)
    top_negative = gain_df.sort_values("primary_gain", ascending=True).head(5)

    lines = []
    lines.append("Pairwise Transfer Hypothesis Analysis")
    lines.append("")
    lines.append(f"Primary gain metric: {gain_metric}")
    lines.append(f"Comparisons analyzed: {len(gain_df)}")
    lines.append(f"Unique pairs: {gain_df['pair_id'].nunique()}")
    lines.append(f"Seeds: {gain_df['seed'].nunique()}")
    lines.append("")
    lines.append("By target fraction:")
    for row in fraction_summary.itertuples(index=False):
        lines.append(
            "  "
            f"frac={row.target_fraction:.2f} | n={row.n_runs} | "
            f"positive={int(row.positive_transfer)} | negative={int(row.negative_transfer)} | "
            f"neutral={int(row.neutral_transfer)} | mean_gain={row.mean_primary_gain:.4f} | "
            f"median_gain={row.median_primary_gain:.4f}"
        )
    lines.append("")
    lines.append("Top positive cases:")
    for _, row in top_positive.iterrows():
        lines.append("  " + format_case_line(row, gain_metric))
    lines.append("")
    lines.append("Top negative cases:")
    for _, row in top_negative.iterrows():
        lines.append("  " + format_case_line(row, gain_metric))

    if predictor_df is not None and not predictor_df.empty:
        lines.append("")
        lines.append("Strongest predictor correlations (overall, by absolute Spearman):")
        overall_df = predictor_df[predictor_df["scope"] == "overall"].head(5)
        for row in overall_df.itertuples(index=False):
            lines.append(
                "  "
                f"{row.predictor} | n={row.n_runs} | "
                f"pearson={row.pearson_corr:.4f} | spearman={row.spearman_corr:.4f}"
            )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    domain_pairs_path = Path(args.domain_pairs)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_df, pairs_df = load_inputs(run_dir=run_dir, domain_pairs_path=domain_pairs_path)
    valid_df = prepare_valid_summary(summary_df=summary_df, pairs_df=pairs_df)
    gain_df = build_gain_table(valid_df=valid_df)

    compatibility_df = compute_compatibility_table(
        gain_df=gain_df,
        run_dir=run_dir,
        top_k_values=sorted(set(args.top_k)),
    )
    if not compatibility_df.empty:
        gain_df = gain_df.merge(
            compatibility_df,
            on=["pair_id", "seed", "target_fraction"],
            how="left",
        )

    gain_df = add_transfer_labels(
        gain_df=gain_df,
        gain_metric=args.gain_metric,
        benefit_margin=args.benefit_margin,
    )

    fraction_summary = summarize_by_fraction(gain_df)
    pair_summary = summarize_by_pair(gain_df)
    metric_gain_summary = summarize_metric_gains(gain_df)
    predictor_df = build_predictor_correlation_table(gain_df)

    gain_df.to_csv(out_dir / "gain_table.csv", index=False)
    fraction_summary.to_csv(out_dir / "fraction_gain_summary.csv", index=False)
    pair_summary.to_csv(out_dir / "pair_gain_summary.csv", index=False)
    metric_gain_summary.to_csv(out_dir / "metric_gain_summary.csv", index=False)
    predictor_df.to_csv(out_dir / "predictor_correlations.csv", index=False)
    gain_df.sort_values("primary_gain", ascending=False).head(20).to_csv(
        out_dir / "top_positive_cases.csv", index=False
    )
    gain_df.sort_values("primary_gain", ascending=True).head(20).to_csv(
        out_dir / "top_negative_cases.csv", index=False
    )
    write_text_summary(
        out_path=out_dir / "analysis_summary.txt",
        gain_df=gain_df,
        fraction_summary=fraction_summary,
        predictor_df=predictor_df,
        gain_metric=args.gain_metric,
    )

    print(f"Saved hypothesis analysis outputs to: {out_dir}")
    print(f"Comparisons analyzed: {len(gain_df)}")
    print(f"Positive transfer cases: {int(gain_df['positive_transfer'].sum())}")
    print(f"Negative transfer cases: {int(gain_df['negative_transfer'].sum())}")


if __name__ == "__main__":
    main()
