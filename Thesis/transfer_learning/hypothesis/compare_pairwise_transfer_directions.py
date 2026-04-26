from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_RUN_DIRS = [
    "transfer_learning/hypothesis/pairwise_runs_iot23_to_unsw_multi_seed",
    "transfer_learning/hypothesis/pairwise_runs_unsw_to_iot23_multi_seed",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine multiple pairwise transfer-hypothesis analysis folders into one "
            "cross-direction comparison."
        )
    )
    parser.add_argument(
        "--run_dirs",
        nargs="+",
        default=DEFAULT_RUN_DIRS,
        help="Run directories that already contain analysis/gain_table.csv.",
    )
    parser.add_argument(
        "--out_dir",
        default="transfer_learning/hypothesis/combined_direction_analysis",
        help="Directory for the combined comparison outputs.",
    )
    parser.add_argument(
        "--gain_metric",
        default="f1_attack",
        help="Primary gain metric label written into the summary text.",
    )
    return parser.parse_args()


def load_gain_table(run_dir: Path) -> pd.DataFrame:
    gain_path = run_dir / "analysis" / "gain_table.csv"
    if not gain_path.exists():
        raise FileNotFoundError(f"Missing gain table: {gain_path}")

    df = pd.read_csv(gain_path)
    df["run_dir"] = str(run_dir)
    df["run_name"] = run_dir.name
    return df


def summarize_by_direction_and_fraction(combined_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        combined_df.groupby(["pair_family", "target_fraction"], dropna=False)
        .agg(
            n_runs=("pair_id", "count"),
            n_pairs=("pair_id", "nunique"),
            n_seeds=("seed", "nunique"),
            positive_transfer=("positive_transfer", "sum"),
            negative_transfer=("negative_transfer", "sum"),
            neutral_transfer=("neutral_transfer", "sum"),
            mean_primary_gain=("primary_gain", "mean"),
            median_primary_gain=("primary_gain", "median"),
            mean_target_only_f1_attack=("target_only_f1_attack", "mean"),
            mean_transfer_f1_attack=("transfer_f1_attack", "mean"),
            mean_source_only_f1_attack=("source_only_f1_attack", "mean"),
        )
        .reset_index()
    )
    summary["positive_rate"] = summary["positive_transfer"] / summary["n_runs"]
    summary["negative_rate"] = summary["negative_transfer"] / summary["n_runs"]
    summary["net_positive_minus_negative"] = summary["positive_transfer"] - summary["negative_transfer"]
    return summary.sort_values(["pair_family", "target_fraction"]).reset_index(drop=True)


def summarize_by_direction(combined_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        combined_df.groupby("pair_family", dropna=False)
        .agg(
            n_runs=("pair_id", "count"),
            n_pairs=("pair_id", "nunique"),
            n_seeds=("seed", "nunique"),
            positive_transfer=("positive_transfer", "sum"),
            negative_transfer=("negative_transfer", "sum"),
            neutral_transfer=("neutral_transfer", "sum"),
            mean_primary_gain=("primary_gain", "mean"),
            median_primary_gain=("primary_gain", "median"),
            min_primary_gain=("primary_gain", "min"),
            max_primary_gain=("primary_gain", "max"),
            mean_target_only_f1_attack=("target_only_f1_attack", "mean"),
            mean_transfer_f1_attack=("transfer_f1_attack", "mean"),
            mean_source_only_f1_attack=("source_only_f1_attack", "mean"),
        )
        .reset_index()
        .sort_values("pair_family")
    )
    summary["positive_rate"] = summary["positive_transfer"] / summary["n_runs"]
    summary["negative_rate"] = summary["negative_transfer"] / summary["n_runs"]
    summary["net_positive_minus_negative"] = summary["positive_transfer"] - summary["negative_transfer"]
    return summary.reset_index(drop=True)


def summarize_pair_stability(combined_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        combined_df.groupby(["pair_family", "pair_id"], dropna=False)
        .agg(
            n_runs=("target_fraction", "count"),
            fractions_seen=("target_fraction", lambda values: "|".join(str(v) for v in sorted(set(values)))),
            mean_primary_gain=("primary_gain", "mean"),
            median_primary_gain=("primary_gain", "median"),
            positive_transfer=("positive_transfer", "sum"),
            negative_transfer=("negative_transfer", "sum"),
            neutral_transfer=("neutral_transfer", "sum"),
            mean_target_only_f1_attack=("target_only_f1_attack", "mean"),
            mean_transfer_f1_attack=("transfer_f1_attack", "mean"),
        )
        .reset_index()
    )
    summary["net_positive_minus_negative"] = summary["positive_transfer"] - summary["negative_transfer"]
    return summary.sort_values(
        ["pair_family", "mean_primary_gain", "net_positive_minus_negative"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def build_case_table(combined_df: pd.DataFrame) -> pd.DataFrame:
    keep_columns = [
        "pair_family",
        "pair_id",
        "seed",
        "target_fraction",
        "primary_gain",
        "transfer_label",
        "target_only_f1_attack",
        "transfer_f1_attack",
        "source_only_f1_attack",
        "abs_attack_rate_gap",
    ]
    keep_columns = [column for column in keep_columns if column in combined_df.columns]
    return combined_df[keep_columns].sort_values(
        ["pair_family", "primary_gain"], ascending=[True, False]
    ).reset_index(drop=True)


def write_text_summary(
    out_path: Path,
    direction_summary: pd.DataFrame,
    fraction_summary: pd.DataFrame,
    combined_df: pd.DataFrame,
    gain_metric: str,
) -> None:
    lines: list[str] = []
    lines.append("Combined Pairwise Transfer Direction Comparison")
    lines.append("")
    lines.append(f"Primary gain metric: {gain_metric}")
    lines.append(f"Total comparisons: {len(combined_df)}")
    lines.append(f"Directions compared: {combined_df['pair_family'].nunique()}")
    lines.append("")
    lines.append("Direction-level summary:")
    for row in direction_summary.itertuples(index=False):
        lines.append(
            "  "
            f"{row.pair_family} | runs={row.n_runs} | pairs={row.n_pairs} | seeds={row.n_seeds} | "
            f"positive={int(row.positive_transfer)} | negative={int(row.negative_transfer)} | "
            f"neutral={int(row.neutral_transfer)} | mean_gain={row.mean_primary_gain:.4f} | "
            f"median_gain={row.median_primary_gain:.4f}"
        )

    lines.append("")
    lines.append("By direction and target fraction:")
    for row in fraction_summary.itertuples(index=False):
        lines.append(
            "  "
            f"{row.pair_family} | frac={row.target_fraction:.2f} | runs={row.n_runs} | "
            f"positive={int(row.positive_transfer)} | negative={int(row.negative_transfer)} | "
            f"neutral={int(row.neutral_transfer)} | mean_gain={row.mean_primary_gain:.4f} | "
            f"median_gain={row.median_primary_gain:.4f}"
        )

    lines.append("")
    lines.append("Most positive cases overall:")
    top_positive = combined_df.sort_values("primary_gain", ascending=False).head(5)
    for row in top_positive.itertuples(index=False):
        lines.append(
            "  "
            f"{row.pair_family} | {row.pair_id} | seed={int(row.seed)} | frac={row.target_fraction:.2f} | "
            f"gain={row.primary_gain:.4f}"
        )

    lines.append("")
    lines.append("Most negative cases overall:")
    top_negative = combined_df.sort_values("primary_gain", ascending=True).head(5)
    for row in top_negative.itertuples(index=False):
        lines.append(
            "  "
            f"{row.pair_family} | {row.pair_id} | seed={int(row.seed)} | frac={row.target_fraction:.2f} | "
            f"gain={row.primary_gain:.4f}"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gain_tables = [load_gain_table(Path(run_dir)) for run_dir in args.run_dirs]
    combined_df = pd.concat(gain_tables, ignore_index=True)

    direction_summary = summarize_by_direction(combined_df)
    fraction_summary = summarize_by_direction_and_fraction(combined_df)
    pair_summary = summarize_pair_stability(combined_df)
    case_table = build_case_table(combined_df)

    combined_df.to_csv(out_dir / "combined_gain_table.csv", index=False)
    direction_summary.to_csv(out_dir / "direction_summary.csv", index=False)
    fraction_summary.to_csv(out_dir / "direction_fraction_summary.csv", index=False)
    pair_summary.to_csv(out_dir / "direction_pair_summary.csv", index=False)
    case_table.to_csv(out_dir / "direction_case_table.csv", index=False)
    write_text_summary(
        out_path=out_dir / "combined_direction_summary.txt",
        direction_summary=direction_summary,
        fraction_summary=fraction_summary,
        combined_df=combined_df,
        gain_metric=args.gain_metric,
    )

    print(f"Saved combined direction analysis to: {out_dir}")
    print(direction_summary.to_string(index=False))


if __name__ == "__main__":
    main()
