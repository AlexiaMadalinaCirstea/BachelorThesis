"""
dataset_statistics.py

Generate dataset-level statistics and thesis-ready plots for the processed IoT-23 dataset.

Expected input:
    <processed_dir>/iot23/all_flows.parquet

Main outputs:
    <processed_dir>/iot23/dataset_statistics/data_summary.csv
    <processed_dir>/iot23/dataset_statistics/scenario_statistics.csv
    <processed_dir>/iot23/dataset_statistics/split_statistics.csv
    <processed_dir>/iot23/dataset_statistics/attack_type_distribution.csv
    <processed_dir>/iot23/dataset_statistics/categorical_distributions.json
    <processed_dir>/iot23/dataset_statistics/plots/*.png

Usage:
    python dataset_statistics.py \
        --processed_dir Datasets/IoT23/processed_full \
        --dataset_name iot23

Example:
    python dataset_statistics.py \
        --processed_dir Datasets/IoT23/processed_full \
        --dataset_name iot23
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s"
)
log = logging.getLogger(__name__)


# Constants


EXPECTED_COLUMNS = [
    "ts",
    "scenario",
    "split",
    "proto",
    "service",
    "conn_state",
    "duration",
    "orig_bytes",
    "resp_bytes",
    "missed_bytes",
    "orig_pkts",
    "orig_ip_bytes",
    "resp_pkts",
    "resp_ip_bytes",
    "bytes_ratio",
    "pkts_ratio",
    "orig_bytes_per_pkt",
    "resp_bytes_per_pkt",
    "label",
    "detailed_label",
    "label_binary",
    "label_phase",
]


# Helpers

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_float(x) -> float:
    if pd.isna(x):
        return 0.0
    return float(x)


def safe_int(x) -> int:
    if pd.isna(x):
        return 0
    return int(x)


def percentage(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator) * 100.0


def save_json(obj: Dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input parquet is missing required columns: {missing}\n"
            "Please run your preprocessing pipeline first."
        )


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Defensive normalization
    df["scenario"] = df["scenario"].astype(str)
    df["split"] = df["split"].astype(str)
    df["label"] = df["label"].astype(str)
    df["detailed_label"] = df["detailed_label"].astype(str)
    df["label_phase"] = df["label_phase"].astype(str)
    df["proto"] = df["proto"].astype(str)
    df["service"] = df["service"].astype(str)
    df["conn_state"] = df["conn_state"].astype(str)

    numeric_cols = [
        "ts",
        "duration",
        "orig_bytes",
        "resp_bytes",
        "missed_bytes",
        "orig_pkts",
        "orig_ip_bytes",
        "resp_pkts",
        "resp_ip_bytes",
        "bytes_ratio",
        "pkts_ratio",
        "orig_bytes_per_pkt",
        "resp_bytes_per_pkt",
        "label_binary",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# Summary builders

def build_data_summary(df: pd.DataFrame) -> pd.DataFrame:
    n_total = len(df)
    n_benign = int((df["label_binary"] == 0).sum())
    n_malicious = int((df["label_binary"] == 1).sum())
    n_scenarios = int(df["scenario"].nunique())
    n_train = int((df["split"] == "train").sum())
    n_val = int((df["split"] == "val").sum())
    n_test = int((df["split"] == "test").sum())
    n_attack_types_detailed = int(
        df.loc[df["label_binary"] == 1, "detailed_label"].replace("unknown", np.nan).dropna().nunique()
    )
    n_attack_types_phase = int(
        df.loc[df["label_binary"] == 1, "label_phase"].replace("unknown", np.nan).dropna().nunique()
    )
    n_zero_duration = int((df["duration"] == 0).sum())
    n_missing_ts = int(df["ts"].isna().sum())

    rows = [
        {"metric": "dataset_name", "value": "iot23"},
        {"metric": "total_flows", "value": n_total},
        {"metric": "total_scenarios", "value": n_scenarios},
        {"metric": "benign_flows", "value": n_benign},
        {"metric": "malicious_flows", "value": n_malicious},
        {"metric": "benign_percentage", "value": round(percentage(n_benign, n_total), 4)},
        {"metric": "malicious_percentage", "value": round(percentage(n_malicious, n_total), 4)},
        {"metric": "train_flows", "value": n_train},
        {"metric": "val_flows", "value": n_val},
        {"metric": "test_flows", "value": n_test},
        {"metric": "train_percentage", "value": round(percentage(n_train, n_total), 4)},
        {"metric": "val_percentage", "value": round(percentage(n_val, n_total), 4)},
        {"metric": "test_percentage", "value": round(percentage(n_test, n_total), 4)},
        {"metric": "unique_attack_types_detailed_label", "value": n_attack_types_detailed},
        {"metric": "unique_attack_types_label_phase", "value": n_attack_types_phase},
        {"metric": "zero_duration_flows", "value": n_zero_duration},
        {"metric": "zero_duration_percentage", "value": round(percentage(n_zero_duration, n_total), 4)},
        {"metric": "missing_timestamps", "value": n_missing_ts},
        {"metric": "missing_timestamp_percentage", "value": round(percentage(n_missing_ts, n_total), 4)},
        {"metric": "duration_mean", "value": round(safe_float(df["duration"].mean()), 6)},
        {"metric": "duration_median", "value": round(safe_float(df["duration"].median()), 6)},
        {"metric": "duration_std", "value": round(safe_float(df["duration"].std()), 6)},
        {"metric": "duration_p95", "value": round(safe_float(df["duration"].quantile(0.95)), 6)},
        {"metric": "duration_p99", "value": round(safe_float(df["duration"].quantile(0.99)), 6)},
    ]
    return pd.DataFrame(rows)


def build_scenario_statistics(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []

    for scenario, g in df.groupby("scenario", sort=True):
        n_total = len(g)
        n_benign = int((g["label_binary"] == 0).sum())
        n_malicious = int((g["label_binary"] == 1).sum())

        malicious_detail = (
            g.loc[g["label_binary"] == 1, "detailed_label"]
            .replace("unknown", np.nan)
            .dropna()
        )

        rows.append(
            {
                "scenario": scenario,
                "split": g["split"].iloc[0] if len(g) > 0 else "unknown",
                "n_flows": n_total,
                "n_benign": n_benign,
                "n_malicious": n_malicious,
                "pct_benign": round(percentage(n_benign, n_total), 4),
                "pct_malicious": round(percentage(n_malicious, n_total), 4),
                "n_unique_attack_types": int(malicious_detail.nunique()),
                "most_common_attack_type": (
                    malicious_detail.value_counts().index[0]
                    if not malicious_detail.empty else "none"
                ),
                "duration_mean": round(safe_float(g["duration"].mean()), 6),
                "duration_median": round(safe_float(g["duration"].median()), 6),
                "duration_p95": round(safe_float(g["duration"].quantile(0.95)), 6),
                "ts_min": safe_float(g["ts"].min()),
                "ts_max": safe_float(g["ts"].max()),
                "n_zero_duration": int((g["duration"] == 0).sum()),
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(["split", "n_flows", "scenario"], ascending=[True, False, True]).reset_index(drop=True)
    return out


def build_split_statistics(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []

    for split, g in df.groupby("split", sort=True):
        n_total = len(g)
        n_benign = int((g["label_binary"] == 0).sum())
        n_malicious = int((g["label_binary"] == 1).sum())

        rows.append(
            {
                "split": split,
                "n_flows": n_total,
                "n_scenarios": int(g["scenario"].nunique()),
                "n_benign": n_benign,
                "n_malicious": n_malicious,
                "pct_benign": round(percentage(n_benign, n_total), 4),
                "pct_malicious": round(percentage(n_malicious, n_total), 4),
                "duration_mean": round(safe_float(g["duration"].mean()), 6),
                "duration_median": round(safe_float(g["duration"].median()), 6),
                "duration_p95": round(safe_float(g["duration"].quantile(0.95)), 6),
            }
        )

    return pd.DataFrame(rows).sort_values("split").reset_index(drop=True)


def build_attack_type_distribution(df: pd.DataFrame) -> pd.DataFrame:
    attack_df = df.loc[df["label_binary"] == 1].copy()

    if attack_df.empty:
        return pd.DataFrame(columns=["detailed_label", "count", "percentage"])

    counts = (
        attack_df["detailed_label"]
        .fillna("unknown")
        .value_counts(dropna=False)
        .rename_axis("detailed_label")
        .reset_index(name="count")
    )
    counts["percentage"] = counts["count"].apply(lambda x: round(percentage(x, len(attack_df)), 4))
    return counts


def build_label_phase_distribution(df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        df["label_phase"]
        .fillna("unknown")
        .value_counts(dropna=False)
        .rename_axis("label_phase")
        .reset_index(name="count")
    )
    counts["percentage"] = counts["count"].apply(lambda x: round(percentage(x, len(df)), 4))
    return counts


def build_categorical_distributions(df: pd.DataFrame, top_n: int = 20) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}

    for col in ["proto", "service", "conn_state"]:
        counts = df[col].fillna("unknown").astype(str).value_counts(dropna=False).head(top_n)
        out[col] = {str(k): int(v) for k, v in counts.to_dict().items()}

    return out


# Plotting

def plot_global_class_distribution(df: pd.DataFrame, out_path: Path) -> None:
    counts = df["label_binary"].map({0: "Benign", 1: "Malicious"}).value_counts()
    order = ["Benign", "Malicious"]
    values = [counts.get(k, 0) for k in order]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(order, values)
    ax.set_title("Global benign vs malicious flow distribution")
    ax.set_ylabel("Number of flows")

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:,}",
            ha="center",
            va="bottom"
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_split_distribution(split_stats: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(split_stats["split"], split_stats["n_flows"])
    ax.set_title("Flow distribution across splits")
    ax.set_ylabel("Number of flows")
    ax.set_xlabel("Split")

    for bar, value in zip(bars, split_stats["n_flows"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{int(value):,}",
            ha="center",
            va="bottom"
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_scenario_flow_counts(scenario_stats: pd.DataFrame, out_path: Path) -> None:
    plot_df = scenario_stats.sort_values("n_flows", ascending=False).reset_index(drop=True)

    fig_height = max(6, 0.35 * len(plot_df))
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.barh(plot_df["scenario"], plot_df["n_flows"])
    ax.invert_yaxis()
    ax.set_title("Number of flows per scenario")
    ax.set_xlabel("Number of flows")
    ax.set_ylabel("Scenario")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_scenario_malicious_percentage(scenario_stats: pd.DataFrame, out_path: Path) -> None:
    plot_df = scenario_stats.sort_values("pct_malicious", ascending=False).reset_index(drop=True)

    fig_height = max(6, 0.35 * len(plot_df))
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.barh(plot_df["scenario"], plot_df["pct_malicious"])
    ax.invert_yaxis()
    ax.set_title("Percentage of malicious flows per scenario")
    ax.set_xlabel("Malicious flows (%)")
    ax.set_ylabel("Scenario")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_attack_type_distribution(attack_stats: pd.DataFrame, out_path: Path, top_n: int = 15) -> None:
    plot_df = attack_stats.head(top_n).copy()

    fig_height = max(5, 0.45 * len(plot_df))
    fig, ax = plt.subplots(figsize=(11, fig_height))
    ax.barh(plot_df["detailed_label"], plot_df["count"])
    ax.invert_yaxis()
    ax.set_title(f"Top {top_n} malicious detailed labels")
    ax.set_xlabel("Number of malicious flows")
    ax.set_ylabel("Detailed label")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_label_phase_distribution(label_phase_stats: pd.DataFrame, out_path: Path) -> None:
    plot_df = label_phase_stats.copy()

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(plot_df["label_phase"], plot_df["count"])
    ax.set_title("Distribution of label phases")
    ax.set_ylabel("Number of flows")
    ax.set_xlabel("Label phase")
    plt.xticks(rotation=30, ha="right")

    for bar, value in zip(bars, plot_df["count"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{int(value):,}",
            ha="center",
            va="bottom",
            fontsize=8
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_duration_histogram(df: pd.DataFrame, out_path: Path) -> None:
    duration = df["duration"].dropna()
    duration = duration[duration >= 0]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(duration, bins=50)
    ax.set_title("Flow duration distribution")
    ax.set_xlabel("Duration")
    ax.set_ylabel("Frequency")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_duration_histogram_log(df: pd.DataFrame, out_path: Path) -> None:
    duration = df["duration"].dropna()
    duration = duration[duration > 0]

    if duration.empty:
        return

    log_duration = np.log10(duration + 1e-9)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(log_duration, bins=50)
    ax.set_title("Log10 flow duration distribution")
    ax.set_xlabel("log10(duration)")
    ax.set_ylabel("Frequency")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_top_category(df: pd.DataFrame, column: str, out_path: Path, top_n: int = 15) -> None:
    counts = (
        df[column]
        .fillna("unknown")
        .astype(str)
        .value_counts(dropna=False)
        .head(top_n)
        .reset_index()
    )
    counts.columns = [column, "count"]

    fig_height = max(5, 0.4 * len(counts))
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.barh(counts[column], counts["count"])
    ax.invert_yaxis()
    ax.set_title(f"Top {top_n} values for {column}")
    ax.set_xlabel("Count")
    ax.set_ylabel(column)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# Main

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dataset statistics and plots for processed IoT-23 data.")
    parser.add_argument(
        "--processed_dir",
        required=True,
        help="Directory that contains the dataset subfolder, e.g. Datasets/IoT23/processed_full"
    )
    parser.add_argument(
        "--dataset_name",
        default="iot23",
        help="Dataset subfolder name inside processed_dir (default: iot23)"
    )
    args = parser.parse_args()

    dataset_dir = Path(args.processed_dir) / args.dataset_name
    input_path = dataset_dir / "all_flows.parquet"
    output_dir = dataset_dir / "dataset_statistics"
    plots_dir = output_dir / "plots"

    ensure_dir(output_dir)
    ensure_dir(plots_dir)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Could not find input file: {input_path}\n"
            "Expected: <processed_dir>/<dataset_name>/all_flows.parquet"
        )

    log.info("Loading %s", input_path)
    df = pd.read_parquet(input_path)
    validate_columns(df)
    df = prepare_dataframe(df)

    log.info("Building summary tables")
    data_summary = build_data_summary(df)
    scenario_stats = build_scenario_statistics(df)
    split_stats = build_split_statistics(df)
    attack_stats = build_attack_type_distribution(df)
    label_phase_stats = build_label_phase_distribution(df)
    categorical_distributions = build_categorical_distributions(df)

    data_summary.to_csv(output_dir / "data_summary.csv", index=False)
    scenario_stats.to_csv(output_dir / "scenario_statistics.csv", index=False)
    split_stats.to_csv(output_dir / "split_statistics.csv", index=False)
    attack_stats.to_csv(output_dir / "attack_type_distribution.csv", index=False)
    label_phase_stats.to_csv(output_dir / "label_phase_distribution.csv", index=False)
    save_json(categorical_distributions, output_dir / "categorical_distributions.json")

    log.info("Generating plots")
    plot_global_class_distribution(df, plots_dir / "global_class_distribution.png")
    plot_split_distribution(split_stats, plots_dir / "split_distribution.png")
    plot_scenario_flow_counts(scenario_stats, plots_dir / "scenario_flow_counts.png")
    plot_scenario_malicious_percentage(scenario_stats, plots_dir / "scenario_malicious_percentage.png")
    plot_attack_type_distribution(attack_stats, plots_dir / "attack_type_distribution_top15.png", top_n=15)
    plot_label_phase_distribution(label_phase_stats, plots_dir / "label_phase_distribution.png")
    plot_duration_histogram(df, plots_dir / "duration_histogram.png")
    plot_duration_histogram_log(df, plots_dir / "duration_histogram_log10.png")
    plot_top_category(df, "proto", plots_dir / "top_proto.png", top_n=10)
    plot_top_category(df, "service", plots_dir / "top_service.png", top_n=15)
    plot_top_category(df, "conn_state", plots_dir / "top_conn_state.png", top_n=15)

    log.info("Done")
    print("\nGenerated outputs:")
    print(f"  Tables: {output_dir}")
    print(f"  Plots : {plots_dir}\n")


if __name__ == "__main__":
    main()