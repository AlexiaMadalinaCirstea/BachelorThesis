#!/usr/bin/env python3
"""
full_iot23_dataset_statistics.py

Compute thesis-ready dataset statistics and cross-scenario analyses for IoT-23
processed per-scenario parquet files.

Main outputs:
- scenario_stats.csv
- attack_type_counts_per_scenario.csv
- global_label_distribution.csv
- numeric_feature_summary_by_scenario.csv
- feature_shift_scores.csv
- scenario_similarity_numeric.csv
- scenario_similarity_labels.csv
- optional scenario_difficulty_analysis.csv

Main plots:
- flows_per_scenario.png
- malicious_ratio_per_scenario.png
- label_composition_per_scenario.png
- global_attack_type_distribution.png
- top_feature_shift_scores.png
- scenario_similarity_numeric.png
- scenario_similarity_labels.png
- normalized_feature_heatmap.png

Optional:
- If you pass a LOSO results CSV with columns [scenario, f1_macro] or similar,
  the script will merge scenario properties with fold difficulty and compute
  simple correlations.

Usage example:
python full_iot23_dataset_statistics.py \
    --input-dir processed_scenarios \
    --output-dir dataset_statistics \
    --top-features-heatmap 20

Optional difficulty merge:
python full_iot23_dataset_statistics.py \
    --input-dir processed_scenarios \
    --output-dir dataset_statistics \
    --difficulty-csv loso_results.csv

Assumptions:
- Each parquet file is one scenario.
- Your processed parquet files contain at least one label column.
- The script will try to auto-detect:
    * a detailed label column (multi-class, e.g. benign / C&C / DDoS / ...)
    * a binary label column if present
- If no binary label exists, it derives one from the detailed label.

Notes:
- This script avoids seaborn.
- It uses matplotlib only.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Helpers

DETAILED_LABEL_CANDIDATES = [
    "label_multi",
    "multi_label",
    "attack_type",
    "phase_label",
    "label_detailed",
    "label_multiclass",
    "class_label",
    "detailed_label",
    "label",
]

BINARY_LABEL_CANDIDATES = [
    "label_binary",
    "binary_label",
    "is_malicious",
    "target",
    "y",
]

IGNORE_NUMERIC_COLUMNS = {
    "ts",
    "timestamp",
    "time",
    "unix_ts",
    "uid_hash",
    "row_id",
    "fold",
    "split",
}

IGNORE_PREFIXES = (
    "label",
    "target",
    "is_",
)


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_text_label(value: object) -> str:
    if pd.isna(value):
        return "unknown"
    s = str(value).strip()
    if s == "":
        return "unknown"
    return s


def is_benign_label(value: object) -> bool:
    s = normalize_text_label(value).lower()
    benign_tokens = {
        "benign",
        "normal",
        "background",
        "legitimate",
        "clean",
        "0",
        "false",
    }
    return s in benign_tokens


def detect_label_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns:
        detailed_label_col, binary_label_col
    """
    cols = list(df.columns)

    detailed = None
    for c in DETAILED_LABEL_CANDIDATES:
        if c in cols:
            detailed = c
            break

    binary = None
    for c in BINARY_LABEL_CANDIDATES:
        if c in cols:
            binary = c
            break

    # If there is no explicit binary label but 'label' looks binary/numeric-ish,
    # let detailed='label' and derive binary later.
    return detailed, binary


def derive_binary_label(df: pd.DataFrame, detailed_col: Optional[str], binary_col: Optional[str]) -> pd.Series:
    """
    Returns a binary malicious indicator: 0=benign, 1=malicious
    """
    if binary_col is not None:
        s = df[binary_col]

        # Numeric 0/1
        if pd.api.types.is_numeric_dtype(s):
            return s.fillna(0).astype(int).clip(lower=0, upper=1)

        # String values
        return s.map(lambda x: 0 if is_benign_label(x) else 1).astype(int)

    if detailed_col is None:
        raise ValueError("No label column could be detected. Please inspect your parquet schema.")

    return df[detailed_col].map(lambda x: 0 if is_benign_label(x) else 1).astype(int)


def derive_detailed_label(df: pd.DataFrame, detailed_col: Optional[str], binary_col: Optional[str]) -> pd.Series:
    """
    Returns a normalized multi-class label series.
    If only binary exists, map to benign/malicious.
    """
    if detailed_col is not None:
        return df[detailed_col].map(normalize_text_label)

    if binary_col is not None:
        s = df[binary_col]
        if pd.api.types.is_numeric_dtype(s):
            return s.map(lambda x: "benign" if int(pd.notna(x) and x == 0) else "malicious")
        return s.map(lambda x: "benign" if is_benign_label(x) else "malicious")

    raise ValueError("No label column could be detected. Please inspect your parquet schema.")


def get_numeric_feature_columns(df: pd.DataFrame, excluded: Iterable[str]) -> List[str]:
    excluded_set = set(excluded)
    numeric_cols: List[str] = []

    for c in df.columns:
        if c in excluded_set:
            continue
        if c.lower() in IGNORE_NUMERIC_COLUMNS:
            continue
        if c.lower().startswith(IGNORE_PREFIXES):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)

    return numeric_cols


def cosine_similarity_matrix(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    Xn = X / norms
    return Xn @ Xn.T


def zscore_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    arr = df.to_numpy(dtype=float)
    means = np.nanmean(arr, axis=0)
    stds = np.nanstd(arr, axis=0)
    stds = np.where(stds == 0, 1.0, stds)
    z = (arr - means) / stds
    return pd.DataFrame(z, index=df.index, columns=df.columns)


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    p = np.maximum(p, eps)
    q = np.maximum(q, eps)

    p = p / p.sum()
    q = q / q.sum()

    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sum(a * np.log2(a / b)))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def pairwise_mean_js_divergence(distributions: np.ndarray) -> float:
    """
    distributions shape: [n_scenarios, n_categories]
    """
    n = distributions.shape[0]
    if n < 2:
        return 0.0

    vals = []
    for i in range(n):
        for j in range(i + 1, n):
            vals.append(js_divergence(distributions[i], distributions[j]))
    return float(np.mean(vals)) if vals else 0.0


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def save_matrix_csv(matrix: pd.DataFrame, path: Path) -> None:
    matrix.to_csv(path)


def plot_bar(
    x: Sequence[str],
    y: Sequence[float],
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    rotation: int = 90,
    logy: bool = False,
    figsize: Tuple[int, int] = (14, 6),
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x, y)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    plt.xticks(rotation=rotation, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_stacked_bar(
    df: pd.DataFrame,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    figsize: Tuple[int, int] = (16, 7),
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    bottom = np.zeros(len(df), dtype=float)

    for col in df.columns:
        vals = df[col].to_numpy(dtype=float)
        ax.bar(df.index, vals, bottom=bottom, label=col)
        bottom += vals

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right", bbox_to_anchor=(1.18, 1.0))
    plt.xticks(rotation=90, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_heatmap(
    matrix: pd.DataFrame,
    title: str,
    out_path: Path,
    cmap: str = "viridis",
    figsize: Tuple[int, int] = (10, 8),
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=90)
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def pick_top_features_by_shift(feature_shift_scores: pd.DataFrame, n: int) -> List[str]:
    if feature_shift_scores.empty:
        return []
    cols = ["feature", "mean_cv_across_scenarios", "scenario_mean_range", "scenario_mean_std"]
    existing = [c for c in cols if c in feature_shift_scores.columns]
    if "mean_cv_across_scenarios" in feature_shift_scores.columns:
        ranked = feature_shift_scores.sort_values("mean_cv_across_scenarios", ascending=False)
    else:
        ranked = feature_shift_scores.sort_values(existing[1], ascending=False) if len(existing) > 1 else feature_shift_scores
    return ranked["feature"].head(n).tolist()


def infer_attack_family(label_name: str) -> str:
    s = normalize_text_label(label_name).lower()

    if s == "benign":
        return "benign"
    if "c&c" in s or "cnc" in s or "command" in s:
        return "c&c"
    if "ddos" in s or "dos" in s:
        return "ddos"
    if "scan" in s or "portscan" in s:
        return "scanning"
    if "filedownload" in s or "download" in s:
        return "filedownload"
    if "attack" in s:
        return "attack"
    return s


def read_difficulty_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols_lower = {c.lower(): c for c in df.columns}

    scenario_col = None
    for cand in ["scenario", "fold", "test_scenario", "held_out_scenario"]:
        if cand in cols_lower:
            scenario_col = cols_lower[cand]
            break

    metric_col = None
    for cand in ["f1_macro", "f1", "f1_score", "loso_f1", "macro_f1"]:
        if cand in cols_lower:
            metric_col = cols_lower[cand]
            break

    if scenario_col is None or metric_col is None:
        raise ValueError(
            "Difficulty CSV must contain a scenario column and an F1-like metric column "
            "(e.g. scenario + f1_macro)."
        )

    out = df[[scenario_col, metric_col]].copy()
    out.columns = ["scenario", "f1_macro"]
    out["difficulty"] = 1.0 - out["f1_macro"].astype(float)
    return out


# Main pipeline

def compute_statistics(
    input_dir: Path,
    output_dir: Path,
    top_features_heatmap: int = 20,
    difficulty_csv: Optional[Path] = None,
) -> None:
    safe_mkdir(output_dir)
    plots_dir = output_dir / "plots"
    safe_mkdir(plots_dir)

    parquet_files = sorted(input_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {input_dir}")

    scenario_rows: List[Dict[str, object]] = []
    attack_type_rows: List[Dict[str, object]] = []
    global_label_counts: Dict[str, int] = {}
    numeric_summary_rows: List[Dict[str, object]] = []

    scenario_feature_profiles: Dict[str, Dict[str, float]] = {}
    scenario_label_profiles: Dict[str, Dict[str, float]] = {}
    all_numeric_features: set[str] = set()
    all_detailed_labels: set[str] = set()

    metadata_rows: List[Dict[str, object]] = []

    print(f"[INFO] Found {len(parquet_files)} scenario parquet files.")

    for parquet_path in parquet_files:
        scenario = parquet_path.stem
        print(f"[INFO] Processing: {scenario}")

        df = pd.read_parquet(parquet_path)
        if df.empty:
            warnings.warn(f"{scenario} is empty. Skipping.")
            continue

        detailed_col, binary_col = detect_label_columns(df)
        detailed_labels = derive_detailed_label(df, detailed_col, binary_col)
        binary_labels = derive_binary_label(df, detailed_col, binary_col)

        temp = df.copy()
        temp["_detailed_label"] = detailed_labels
        temp["_binary_label"] = binary_labels

        total_flows = len(temp)
        benign_flows = int((temp["_binary_label"] == 0).sum())
        malicious_flows = int((temp["_binary_label"] == 1).sum())
        malicious_ratio = malicious_flows / total_flows if total_flows > 0 else 0.0
        benign_ratio = benign_flows / total_flows if total_flows > 0 else 0.0

        label_counts = temp["_detailed_label"].value_counts(dropna=False).sort_values(ascending=False)
        non_benign_counts = label_counts[label_counts.index.map(lambda x: not is_benign_label(x))]
        dominant_attack_type = non_benign_counts.index[0] if not non_benign_counts.empty else "none"
        num_attack_types = int(len(non_benign_counts.index))

        attack_family_counts = (
            pd.Series(label_counts.index)
            .map(infer_attack_family)
            .value_counts()
            .to_dict()
        )

        scenario_rows.append(
            {
                "scenario": scenario,
                "total_flows": total_flows,
                "benign_flows": benign_flows,
                "malicious_flows": malicious_flows,
                "benign_ratio": benign_ratio,
                "malicious_ratio": malicious_ratio,
                "num_unique_labels": int(label_counts.shape[0]),
                "num_attack_types": num_attack_types,
                "dominant_attack_type": dominant_attack_type,
                "dominant_attack_type_count": int(non_benign_counts.iloc[0]) if not non_benign_counts.empty else 0,
                "dominant_attack_type_ratio_within_scenario": (
                    float(non_benign_counts.iloc[0] / total_flows) if not non_benign_counts.empty else 0.0
                ),
            }
        )

        metadata_rows.append(
            {
                "scenario": scenario,
                "detected_detailed_label_column": detailed_col or "",
                "detected_binary_label_column": binary_col or "",
            }
        )

        for label_name, count in label_counts.items():
            attack_type_rows.append(
                {
                    "scenario": scenario,
                    "label": label_name,
                    "attack_family": infer_attack_family(label_name),
                    "flow_count": int(count),
                    "ratio_within_scenario": float(count / total_flows),
                }
            )
            global_label_counts[label_name] = global_label_counts.get(label_name, 0) + int(count)
            all_detailed_labels.add(label_name)

        # Numeric feature stats
        excluded_cols = set(temp.columns)
        excluded_cols.remove("_detailed_label")
        excluded_cols.remove("_binary_label")
        numeric_cols = get_numeric_feature_columns(temp, excluded=["_detailed_label", "_binary_label"])
        all_numeric_features.update(numeric_cols)

        feature_profile_for_similarity: Dict[str, float] = {}

        for col in numeric_cols:
            series = pd.to_numeric(temp[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            non_null = series.dropna()
            if non_null.empty:
                continue

            mean_val = float(non_null.mean())
            std_val = float(non_null.std(ddof=0))
            median_val = float(non_null.median())
            min_val = float(non_null.min())
            max_val = float(non_null.max())
            q25 = float(non_null.quantile(0.25))
            q75 = float(non_null.quantile(0.75))
            p95 = float(non_null.quantile(0.95))
            cv = float(std_val / abs(mean_val)) if mean_val != 0 else math.nan

            numeric_summary_rows.append(
                {
                    "scenario": scenario,
                    "feature": col,
                    "non_null_count": int(non_null.shape[0]),
                    "mean": mean_val,
                    "std": std_val,
                    "cv": cv,
                    "median": median_val,
                    "min": min_val,
                    "q25": q25,
                    "q75": q75,
                    "p95": p95,
                    "max": max_val,
                }
            )

            # For cross-scenario numeric similarity, use scenario-level feature means.
            feature_profile_for_similarity[col] = mean_val

        scenario_feature_profiles[scenario] = feature_profile_for_similarity

        # Label-profile proportions per scenario
        scenario_label_profile: Dict[str, float] = {}
        for label_name, count in label_counts.items():
            scenario_label_profile[label_name] = float(count / total_flows)
        scenario_label_profiles[scenario] = scenario_label_profile

    # Save scenario-level tables

    scenario_stats = pd.DataFrame(scenario_rows).sort_values("total_flows", ascending=False).reset_index(drop=True)
    attack_type_counts = pd.DataFrame(attack_type_rows).sort_values(["scenario", "flow_count"], ascending=[True, False]).reset_index(drop=True)
    metadata_df = pd.DataFrame(metadata_rows)

    global_label_distribution = (
        pd.DataFrame(
            [{"label": k, "flow_count": v, "attack_family": infer_attack_family(k)} for k, v in global_label_counts.items()]
        )
        .sort_values("flow_count", ascending=False)
        .reset_index(drop=True)
    )
    global_total_flows = int(global_label_distribution["flow_count"].sum()) if not global_label_distribution.empty else 0
    if global_total_flows > 0:
        global_label_distribution["global_ratio"] = global_label_distribution["flow_count"] / global_total_flows
    else:
        global_label_distribution["global_ratio"] = 0.0

    numeric_feature_summary = pd.DataFrame(numeric_summary_rows)

    save_dataframe(scenario_stats, output_dir / "scenario_stats.csv")
    save_dataframe(attack_type_counts, output_dir / "attack_type_counts_per_scenario.csv")
    save_dataframe(global_label_distribution, output_dir / "global_label_distribution.csv")
    save_dataframe(numeric_feature_summary, output_dir / "numeric_feature_summary_by_scenario.csv")
    save_dataframe(metadata_df, output_dir / "column_detection_metadata.csv")

    # Feature shift scores

    feature_shift_rows: List[Dict[str, object]] = []

    if not numeric_feature_summary.empty:
        grouped = numeric_feature_summary.groupby("feature", dropna=False)

        for feature, g in grouped:
            scenario_means = g["mean"].astype(float).to_numpy()
            scenario_stds = g["std"].astype(float).to_numpy()
            scenario_medians = g["median"].astype(float).to_numpy()

            mean_of_means = float(np.nanmean(scenario_means))
            std_of_means = float(np.nanstd(scenario_means))
            mean_cv = float(std_of_means / abs(mean_of_means)) if mean_of_means != 0 else math.nan

            scenario_mean_range = float(np.nanmax(scenario_means) - np.nanmin(scenario_means))
            scenario_mean_iqr = float(np.nanpercentile(scenario_means, 75) - np.nanpercentile(scenario_means, 25))
            scenario_mean_median = float(np.nanmedian(scenario_means))
            scenario_std_mean = float(np.nanmean(scenario_stds))
            scenario_median_std = float(np.nanstd(scenario_medians))

            feature_shift_rows.append(
                {
                    "feature": feature,
                    "n_scenarios_present": int(g.shape[0]),
                    "mean_of_scenario_means": mean_of_means,
                    "scenario_mean_std": std_of_means,
                    "mean_cv_across_scenarios": mean_cv,
                    "scenario_mean_range": scenario_mean_range,
                    "scenario_mean_iqr": scenario_mean_iqr,
                    "scenario_median_std": scenario_median_std,
                    "mean_of_within_scenario_std": scenario_std_mean,
                }
            )

    feature_shift_scores = pd.DataFrame(feature_shift_rows)
    if not feature_shift_scores.empty:
        feature_shift_scores = feature_shift_scores.sort_values(
            ["mean_cv_across_scenarios", "scenario_mean_std"],
            ascending=[False, False],
            na_position="last",
        ).reset_index(drop=True)

    save_dataframe(feature_shift_scores, output_dir / "feature_shift_scores.csv")

    # Scenario similarity: numeric profiles

    scenarios_sorted = sorted(scenario_feature_profiles.keys())
    numeric_similarity_df = pd.DataFrame()
    normalized_feature_heatmap_df = pd.DataFrame()

    if scenarios_sorted and all_numeric_features:
        all_numeric_features_sorted = sorted(all_numeric_features)

        numeric_profile_matrix = []
        for sc in scenarios_sorted:
            row = [scenario_feature_profiles[sc].get(f, 0.0) for f in all_numeric_features_sorted]
            numeric_profile_matrix.append(row)

        numeric_profile_df = pd.DataFrame(
            numeric_profile_matrix,
            index=scenarios_sorted,
            columns=all_numeric_features_sorted,
        )

        # Log1p on non-negative values where possible to reduce extreme scale.
        transformed = numeric_profile_df.copy()
        for c in transformed.columns:
            vals = transformed[c].to_numpy(dtype=float)
            if np.all(np.isfinite(vals)) and np.nanmin(vals) >= 0:
                transformed[c] = np.log1p(vals)

        # Z-score by feature for comparability
        normalized_feature_heatmap_df = zscore_dataframe(transformed.fillna(0.0))

        sim = cosine_similarity_matrix(normalized_feature_heatmap_df.to_numpy(dtype=float))
        numeric_similarity_df = pd.DataFrame(sim, index=scenarios_sorted, columns=scenarios_sorted)

        save_matrix_csv(numeric_similarity_df, output_dir / "scenario_similarity_numeric.csv")
        normalized_feature_heatmap_df.to_csv(output_dir / "normalized_scenario_feature_profiles.csv")

   
    # Scenario similarity: label profiles

    label_similarity_df = pd.DataFrame()
    label_shift_summary_df = pd.DataFrame()

    if scenario_label_profiles and all_detailed_labels:
        labels_sorted = sorted(all_detailed_labels)
        label_profile_matrix = []
        for sc in scenarios_sorted:
            row = [scenario_label_profiles[sc].get(lbl, 0.0) for lbl in labels_sorted]
            label_profile_matrix.append(row)

        label_profile_df = pd.DataFrame(
            label_profile_matrix,
            index=scenarios_sorted,
            columns=labels_sorted,
        )

        sim = cosine_similarity_matrix(label_profile_df.to_numpy(dtype=float))
        label_similarity_df = pd.DataFrame(sim, index=scenarios_sorted, columns=scenarios_sorted)
        save_matrix_csv(label_similarity_df, output_dir / "scenario_similarity_labels.csv")
        label_profile_df.to_csv(output_dir / "scenario_label_profiles.csv")

        # Per-label cross-scenario variability
        label_shift_rows = []
        for lbl in labels_sorted:
            vals = label_profile_df[lbl].to_numpy(dtype=float)
            mean_ratio = float(np.mean(vals))
            std_ratio = float(np.std(vals))
            cv_ratio = float(std_ratio / abs(mean_ratio)) if mean_ratio != 0 else math.nan
            label_shift_rows.append(
                {
                    "label": lbl,
                    "attack_family": infer_attack_family(lbl),
                    "mean_ratio_across_scenarios": mean_ratio,
                    "std_ratio_across_scenarios": std_ratio,
                    "cv_ratio_across_scenarios": cv_ratio,
                    "max_ratio": float(np.max(vals)),
                    "min_ratio": float(np.min(vals)),
                    "range_ratio": float(np.max(vals) - np.min(vals)),
                }
            )

        label_shift_summary_df = pd.DataFrame(label_shift_rows).sort_values(
            ["cv_ratio_across_scenarios", "std_ratio_across_scenarios"],
            ascending=[False, False],
            na_position="last",
        )
        save_dataframe(label_shift_summary_df, output_dir / "label_shift_scores.csv")

        # Mean pairwise JS divergence across scenarios
        js_value = pairwise_mean_js_divergence(label_profile_df.to_numpy(dtype=float))
        with open(output_dir / "label_profile_shift_summary.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "mean_pairwise_js_divergence": js_value,
                    "n_scenarios": len(label_profile_df),
                    "n_labels": len(label_profile_df.columns),
                },
                f,
                indent=2,
            )

   
    # An optional difficulty analysis

    if difficulty_csv is not None:
        difficulty_df = read_difficulty_csv(difficulty_csv)
        merged = scenario_stats.merge(difficulty_df, on="scenario", how="inner")

        corr_rows = []
        for col in [
            "total_flows",
            "benign_flows",
            "malicious_flows",
            "benign_ratio",
            "malicious_ratio",
            "num_unique_labels",
            "num_attack_types",
            "dominant_attack_type_ratio_within_scenario",
        ]:
            if col in merged.columns and merged[col].notna().sum() > 1:
                corr = merged[[col, "difficulty"]].corr(method="spearman").iloc[0, 1]
                corr_rows.append({"variable": col, "spearman_corr_with_difficulty": corr})

        corr_df = pd.DataFrame(corr_rows).sort_values(
            "spearman_corr_with_difficulty",
            key=lambda s: s.abs(),
            ascending=False,
        )
        save_dataframe(merged, output_dir / "scenario_difficulty_analysis.csv")
        save_dataframe(corr_df, output_dir / "scenario_difficulty_correlations.csv")

    # Plots
 
    if not scenario_stats.empty:
        s1 = scenario_stats.sort_values("total_flows", ascending=False)
        plot_bar(
            x=s1["scenario"].tolist(),
            y=s1["total_flows"].tolist(),
            title="IoT-23 Flows per Scenario",
            xlabel="Scenario",
            ylabel="Number of flows",
            out_path=plots_dir / "flows_per_scenario.png",
            logy=True,
        )

        s2 = scenario_stats.sort_values("malicious_ratio", ascending=False)
        plot_bar(
            x=s2["scenario"].tolist(),
            y=s2["malicious_ratio"].tolist(),
            title="IoT-23 Malicious Ratio per Scenario",
            xlabel="Scenario",
            ylabel="Malicious flow ratio",
            out_path=plots_dir / "malicious_ratio_per_scenario.png",
            logy=False,
        )

    if not attack_type_counts.empty:
        # Stacked label composition per scenario
        comp = attack_type_counts.pivot_table(
            index="scenario",
            columns="label",
            values="ratio_within_scenario",
            aggfunc="sum",
            fill_value=0.0,
        )
        comp = comp.loc[scenario_stats["scenario"].tolist()] if not scenario_stats.empty else comp
        plot_stacked_bar(
            df=comp,
            title="Label Composition per Scenario",
            xlabel="Scenario",
            ylabel="Within-scenario ratio",
            out_path=plots_dir / "label_composition_per_scenario.png",
        )

    if not global_label_distribution.empty:
        plot_bar(
            x=global_label_distribution["label"].tolist(),
            y=global_label_distribution["flow_count"].tolist(),
            title="Global Label Distribution Across IoT-23",
            xlabel="Label",
            ylabel="Flow count",
            out_path=plots_dir / "global_attack_type_distribution.png",
            rotation=45,
            logy=True,
            figsize=(12, 6),
        )

    if not feature_shift_scores.empty:
        top_shift = feature_shift_scores.head(20)
        plot_bar(
            x=top_shift["feature"].tolist(),
            y=top_shift["mean_cv_across_scenarios"].fillna(0.0).tolist(),
            title="Top Shifted Numeric Features Across Scenarios",
            xlabel="Feature",
            ylabel="CV of scenario means",
            out_path=plots_dir / "top_feature_shift_scores.png",
            rotation=75,
            figsize=(14, 6),
        )

    if not numeric_similarity_df.empty:
        plot_heatmap(
            matrix=numeric_similarity_df,
            title="Scenario Similarity Heatmap (Numeric Feature Profiles)",
            out_path=plots_dir / "scenario_similarity_numeric.png",
            cmap="viridis",
            figsize=(12, 10),
            vmin=0.0,
            vmax=1.0,
        )

    if not label_similarity_df.empty:
        plot_heatmap(
            matrix=label_similarity_df,
            title="Scenario Similarity Heatmap (Label Profiles)",
            out_path=plots_dir / "scenario_similarity_labels.png",
            cmap="viridis",
            figsize=(12, 10),
            vmin=0.0,
            vmax=1.0,
        )

    if not normalized_feature_heatmap_df.empty:
        top_features = pick_top_features_by_shift(feature_shift_scores, top_features_heatmap)
        if top_features:
            heat_df = normalized_feature_heatmap_df[top_features]
        else:
            heat_df = normalized_feature_heatmap_df.iloc[:, : min(top_features_heatmap, normalized_feature_heatmap_df.shape[1])]

        plot_heatmap(
            matrix=heat_df,
            title="Normalized Scenario-Feature Heatmap (Top Shifted Features)",
            out_path=plots_dir / "normalized_feature_heatmap.png",
            cmap="coolwarm",
            figsize=(14, 10),
        )

    # Short machine-readable summary
  
    summary = {
        "n_scenarios_processed": int(len(scenario_stats)),
        "total_flows_all_scenarios": int(scenario_stats["total_flows"].sum()) if not scenario_stats.empty else 0,
        "total_benign_flows": int(scenario_stats["benign_flows"].sum()) if not scenario_stats.empty else 0,
        "total_malicious_flows": int(scenario_stats["malicious_flows"].sum()) if not scenario_stats.empty else 0,
        "max_scenario_flows": int(scenario_stats["total_flows"].max()) if not scenario_stats.empty else 0,
        "min_scenario_flows": int(scenario_stats["total_flows"].min()) if not scenario_stats.empty else 0,
        "mean_malicious_ratio": float(scenario_stats["malicious_ratio"].mean()) if not scenario_stats.empty else 0.0,
        "std_malicious_ratio": float(scenario_stats["malicious_ratio"].std(ddof=0)) if not scenario_stats.empty else 0.0,
        "n_unique_detailed_labels": int(global_label_distribution["label"].nunique()) if not global_label_distribution.empty else 0,
    }

    with open(output_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[INFO] Done. Outputs saved to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute full IoT-23 dataset statistics from per-scenario parquet files.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing per-scenario parquet files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where CSVs and plots will be saved.",
    )
    parser.add_argument(
        "--top-features-heatmap",
        type=int,
        default=20,
        help="Number of top shifted features to include in the normalized feature heatmap.",
    )
    parser.add_argument(
        "--difficulty-csv",
        type=Path,
        default=None,
        help="Optional CSV with per-scenario LOSO performance for difficulty analysis.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compute_statistics(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        top_features_heatmap=args.top_features_heatmap,
        difficulty_csv=args.difficulty_csv,
    )


if __name__ == "__main__":
    main()