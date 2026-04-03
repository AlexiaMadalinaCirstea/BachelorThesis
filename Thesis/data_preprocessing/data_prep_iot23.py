"""
IoT-23 preprocessing pipeline (chunked full-dataset-safe version)

What this script does:
- parses all conn.log.labeled files from IoT-23
- extracts Zeek flow features
- assigns binary and multi-phase labels
- applies scenario-level splits (no row-level leakage)
- validates assumptions aggressively
- saves per-scenario chunk parquet files first
- combines chunk files into final scenario parquet
- combines final scenario parquet files into dataset-level outputs

Usage:
    python data_prep_iot23.py \
        --data_dir /path/to/iot_23_datasets_full \
        --out_dir /path/to/processed \
        --sample 1.0 \
        --seed 42

Outputs in <out_dir>/iot23:
    processed_chunks/<scenario>_partXXXX.parquet
    processed_scenarios/<scenario>.parquet
    all_flows.parquet
    train.parquet
    val.parquet
    test.parquet
    label_stats.csv
    feature_info.json
    scenario_summary.csv
    split_summary.json
    data_quality_report.json
    label_mapping_report.csv
"""

from __future__ import annotations

import os
import re
import json
import glob
import gc
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Iterator

import numpy as np
import pandas as pd


# Logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# Constants


ZEEK_NUMERIC_COLS = [
    "duration",
    "orig_bytes",
    "resp_bytes",
    "missed_bytes",
    "orig_pkts",
    "orig_ip_bytes",
    "resp_pkts",
    "resp_ip_bytes",
]

ZEEK_CATEGORICAL_COLS = [
    "proto",
    "service",
    "conn_state",
]

ENGINEERED_COLS = [
    "bytes_ratio",
    "pkts_ratio",
    "orig_bytes_per_pkt",
    "resp_bytes_per_pkt",
]

ZEEK_BASE_COLS = [
    "ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p",
    "proto", "service", "duration", "orig_bytes", "resp_bytes",
    "conn_state", "local_orig", "local_resp", "missed_bytes", "history",
    "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes",
    "tunnel_parents",
]

FINAL_FEATURE_COLS = (
    ["ts", "scenario", "split"]
    + ZEEK_CATEGORICAL_COLS
    + ZEEK_NUMERIC_COLS
    + ENGINEERED_COLS
    + ["label", "detailed_label", "label_binary", "label_phase"]
)

REQUIRED_PARSED_COLS = set(ZEEK_BASE_COLS + ["label", "detailed_label", "scenario"])

ALLOWED_PHASES = {
    "benign",
    "c2c",
    "scanning",
    "ddos",
    "filedownload",
    "attack",
    "other_malicious",
    "unknown",
}

PHASE_MAP = {
    "benign": "benign",
    "c&c": "c2c",
    "c&c-filedownload": "c2c",
    "c&c-heartbeat": "c2c",
    "c&c-heartbeat-attack": "c2c",
    "c&c-heartbeat-filedownload": "c2c",
    "c&c-mirai": "c2c",
    "c&c-partofahorizontalportscan": "c2c",
    "c&c-torii": "c2c",
    "filedownload": "filedownload",
    "heartbeat": "c2c",
    "ddos": "ddos",
    "attack": "attack",
    "okiru": "scanning",
    "okiru-attack": "attack",
    "partofahorizontalportscan": "scanning",
    "partofahorizontalportscan-attack": "attack",
    "mirai": "scanning",
}

TEST_SCENARIOS = [
    "CTU-IoT-Malware-Capture-34-1",
    "CTU-IoT-Malware-Capture-43-1",
    "CTU-IoT-Malware-Capture-44-1",
    "CTU-Honeypot-Capture-4-1",
    "CTU-Honeypot-Capture-5-1",
    "CTU-Honeypot-Capture-7-1",
]

VAL_SCENARIOS = [
    "CTU-IoT-Malware-Capture-49-1",
    "CTU-IoT-Malware-Capture-52-1",
    "CTU-IoT-Malware-Capture-20-1",
]


# Helpers

def save_json(obj: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def safe_int(x) -> int:
    return int(x) if pd.notna(x) else 0


def safe_float(x) -> float:
    return float(x) if pd.notna(x) else 0.0


def scenario_split_lookup() -> Dict[str, str]:
    mapping = {}
    for s in TEST_SCENARIOS:
        mapping[s] = "test"
    for s in VAL_SCENARIOS:
        if s in mapping:
            raise ValueError(f"Scenario appears in both TEST and VAL: {s}")
        mapping[s] = "val"
    return mapping


def validate_static_split_config() -> None:
    test_set = set(TEST_SCENARIOS)
    val_set = set(VAL_SCENARIOS)

    overlap = test_set & val_set
    if overlap:
        raise AssertionError(f"Scenario overlap between VAL and TEST: {sorted(overlap)}")

    if len(test_set) != len(TEST_SCENARIOS):
        raise AssertionError("Duplicate scenario names found in TEST_SCENARIOS.")
    if len(val_set) != len(VAL_SCENARIOS):
        raise AssertionError("Duplicate scenario names found in VAL_SCENARIOS.")


def get_split(scenario_name: str) -> str:
    mapping = scenario_split_lookup()
    return mapping.get(scenario_name, "train")


# Parser helpers

def split_tunnel_and_labels(raw_tail: str) -> Tuple[str, str, str]:
    if raw_tail is None:
        return "-", "-", "-"

    tail_parts = [p.strip() for p in re.split(r"\s{3,}", str(raw_tail).strip()) if p.strip()]

    tunnel_parents = "-"
    label = "-"
    detailed_label = "-"

    if len(tail_parts) >= 1:
        tunnel_parents = tail_parts[0]
    if len(tail_parts) >= 2:
        label = tail_parts[1]
    if len(tail_parts) >= 3:
        detailed_label = tail_parts[2]

    return tunnel_parents, label, detailed_label


def iter_conn_log_chunks(
    filepath: str,
    scenario_name: str,
    chunk_rows: int = 500_000,
    log_every_lines: int = 500_000,
) -> Iterator[Tuple[pd.DataFrame, int, int]]:
    """
    Yield parsed DataFrame chunks for one conn.log.labeled file.

    Yields:
        (chunk_df, total_lines_seen, malformed_rows_so_far)
    """
    rows = []
    malformed_rows = 0
    discovered_fields = None
    line_no = 0

    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh, start=1):
            if line_no % log_every_lines == 0:
                log.info("%s: processed %d lines so far", scenario_name, line_no)

            line = line.rstrip("\n")

            if not line.strip():
                continue

            if line.startswith("#fields"):
                discovered_fields = line.split("\t")[1:]
                continue

            if line.startswith("#"):
                continue

            parts = line.split("\t")

            if len(parts) < 21:
                malformed_rows += 1
                continue

            tunnel_parents, label, detailed_label = split_tunnel_and_labels(parts[20])
            row = parts[:20] + [tunnel_parents, label, detailed_label]

            if len(row) != 23:
                malformed_rows += 1
                continue

            rows.append(row)

            if len(rows) >= chunk_rows:
                all_cols = ZEEK_BASE_COLS + ["label", "detailed_label"]
                chunk_df = pd.DataFrame(rows, columns=all_cols)
                chunk_df["scenario"] = scenario_name
                yield chunk_df, line_no, malformed_rows
                rows = []

        # final remainder
        if rows:
            all_cols = ZEEK_BASE_COLS + ["label", "detailed_label"]
            chunk_df = pd.DataFrame(rows, columns=all_cols)
            chunk_df["scenario"] = scenario_name
            yield chunk_df, line_no, malformed_rows

    if discovered_fields is not None and len(discovered_fields) < 21:
        log.warning(
            "%s: #fields header had only %d fields; expected at least 21",
            scenario_name,
            len(discovered_fields),
        )

    log.info(
        "%s: finished reading %d lines, malformed=%d",
        scenario_name,
        line_no,
        malformed_rows,
    )


# Cleaning / features / labels

def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.replace({"-": np.nan, "(empty)": np.nan})

    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")

    for col in ZEEK_NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df[ZEEK_NUMERIC_COLS] = df[ZEEK_NUMERIC_COLS].fillna(0.0)

    df["bytes_ratio"] = df["orig_bytes"] / (df["resp_bytes"] + 1.0)
    df["pkts_ratio"] = df["orig_pkts"] / (df["resp_pkts"] + 1.0)
    df["orig_bytes_per_pkt"] = df["orig_bytes"] / (df["orig_pkts"] + 1.0)
    df["resp_bytes_per_pkt"] = df["resp_bytes"] / (df["resp_pkts"] + 1.0)

    for col in ENGINEERED_COLS:
        df[col] = df[col].clip(lower=0.0, upper=1e6)

    for col in ZEEK_CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype(str)

    return df


def assign_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    raw_label = (
        df["label"]
        .astype(str)
        .str.lower()
        .str.strip()
        .replace("nan", "unknown")
        .fillna("unknown")
    )

    detailed = (
        df["detailed_label"]
        .astype(str)
        .str.lower()
        .str.strip()
        .replace("nan", "unknown")
        .fillna("unknown")
    )

    df["label"] = raw_label
    df["detailed_label"] = detailed
    df["label_binary"] = raw_label.apply(lambda x: 0 if x == "benign" else 1)
    df["label_phase"] = detailed.apply(
        lambda x: PHASE_MAP.get(x, "other_malicious" if x not in ("benign", "unknown") else x)
    )

    return df


# Validation and audits

def validate_per_scenario_df(df: pd.DataFrame, scenario_name: str) -> None:
    missing_required = REQUIRED_PARSED_COLS - set(df.columns)
    if missing_required:
        raise AssertionError(f"{scenario_name}: missing required columns {sorted(missing_required)}")

    if df["scenario"].nunique() != 1 or df["scenario"].iloc[0] != scenario_name:
        raise AssertionError(f"{scenario_name}: scenario column is inconsistent.")

    invalid_binary = set(df["label_binary"].dropna().unique()) - {0, 1}
    if invalid_binary:
        raise AssertionError(f"{scenario_name}: invalid label_binary values {sorted(invalid_binary)}")

    invalid_phase = set(df["label_phase"].dropna().unique()) - ALLOWED_PHASES
    if invalid_phase:
        raise AssertionError(f"{scenario_name}: invalid label_phase values {sorted(invalid_phase)}")

    if df.empty:
        raise AssertionError(f"{scenario_name}: scenario dataframe is empty after processing.")


def stratified_sample_df(df: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    if frac >= 1.0:
        return df.copy()

    sampled_parts = []
    for _, group in df.groupby("label_binary", dropna=False):
        if len(group) == 0:
            continue
        n = max(1, int(round(len(group) * frac)))
        n = min(n, len(group))
        sampled_parts.append(group.sample(n=n, random_state=seed))

    if not sampled_parts:
        raise AssertionError("Sampling produced no rows.")

    out = pd.concat(sampled_parts, ignore_index=True)
    if out.empty:
        raise AssertionError("Sampling produced an empty dataframe.")

    return out


def validate_combined_df(combined: pd.DataFrame, discovered_scenarios: List[str], strict_benign_train: bool) -> None:
    if combined.empty:
        raise AssertionError("Combined dataset is empty.")

    invalid_binary = set(combined["label_binary"].dropna().unique()) - {0, 1}
    if invalid_binary:
        raise AssertionError(f"Combined data has invalid label_binary values: {sorted(invalid_binary)}")

    invalid_phase = set(combined["label_phase"].dropna().unique()) - ALLOWED_PHASES
    if invalid_phase:
        raise AssertionError(f"Combined data has invalid label_phase values: {sorted(invalid_phase)}")

    scenario_to_split_counts = combined.groupby("scenario")["split"].nunique()
    bad = scenario_to_split_counts[scenario_to_split_counts != 1]
    if not bad.empty:
        raise AssertionError(f"Some scenarios appear in multiple splits: {bad.to_dict()}")

    discovered_set = set(discovered_scenarios)
    combined_set = set(combined["scenario"].unique())
    if discovered_set != combined_set:
        missing = sorted(discovered_set - combined_set)
        extra = sorted(combined_set - discovered_set)
        raise AssertionError(
            f"Mismatch between discovered and combined scenarios. Missing={missing}, Extra={extra}"
        )

    benign_by_split = combined.groupby("split")["label_binary"].apply(lambda s: int((s == 0).sum())).to_dict()
    if benign_by_split.get("train", 0) == 0:
        msg = "Train split contains zero benign rows. This is dangerous for binary IDS training."
        if strict_benign_train:
            raise AssertionError(msg)
        log.warning(msg)


def build_scenario_summary(combined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario, g in combined.groupby("scenario"):
        split = g["split"].iloc[0]
        rows.append({
            "scenario": scenario,
            "split": split,
            "n_rows": len(g),
            "n_benign": int((g["label_binary"] == 0).sum()),
            "n_malicious": int((g["label_binary"] == 1).sum()),
            "n_unknown_ts": int(g["ts"].isna().sum()),
            "n_zero_duration": int((g["duration"] == 0).sum()),
            "n_unknown_label": int((g["label"] == "unknown").sum()),
            "n_unknown_detailed_label": int((g["detailed_label"] == "unknown").sum()),
            "n_other_malicious": int((g["label_phase"] == "other_malicious").sum()),
            "ts_min": safe_float(g["ts"].min()),
            "ts_max": safe_float(g["ts"].max()),
        })
    return pd.DataFrame(rows).sort_values(["split", "scenario"]).reset_index(drop=True)


def build_label_mapping_report(combined: pd.DataFrame) -> pd.DataFrame:
    report = (
        combined.groupby(["label", "detailed_label", "label_phase", "label_binary"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    return report


def build_split_summary(combined: pd.DataFrame) -> dict:
    summary = {}
    for split_name in ["train", "val", "test"]:
        g = combined[combined["split"] == split_name]
        summary[split_name] = {
            "n_rows": int(len(g)),
            "n_scenarios": int(g["scenario"].nunique()),
            "scenarios": sorted(g["scenario"].unique().tolist()),
            "label_binary_counts": {str(k): int(v) for k, v in g["label_binary"].value_counts(dropna=False).to_dict().items()},
            "label_phase_counts": {str(k): int(v) for k, v in g["label_phase"].value_counts(dropna=False).to_dict().items()},
        }
    return summary


def build_data_quality_report(combined: pd.DataFrame) -> dict:
    report = {
        "n_rows_total": int(len(combined)),
        "n_scenarios_total": int(combined["scenario"].nunique()),
        "n_missing_ts": int(combined["ts"].isna().sum()),
        "n_zero_duration": int((combined["duration"] == 0).sum()),
        "n_unknown_label": int((combined["label"] == "unknown").sum()),
        "n_unknown_detailed_label": int((combined["detailed_label"] == "unknown").sum()),
        "n_other_malicious": int((combined["label_phase"] == "other_malicious").sum()),
        "n_all_zero_traffic_rows": int(
            (
                (combined["orig_bytes"] == 0) &
                (combined["resp_bytes"] == 0) &
                (combined["orig_pkts"] == 0) &
                (combined["resp_pkts"] == 0)
            ).sum()
        ),
        "duration_quantiles": {
            "p00": safe_float(combined["duration"].quantile(0.00)),
            "p25": safe_float(combined["duration"].quantile(0.25)),
            "p50": safe_float(combined["duration"].quantile(0.50)),
            "p75": safe_float(combined["duration"].quantile(0.75)),
            "p95": safe_float(combined["duration"].quantile(0.95)),
            "p99": safe_float(combined["duration"].quantile(0.99)),
            "p100": safe_float(combined["duration"].quantile(1.00)),
        },
        "categorical_cardinality": {
            col: int(combined[col].nunique(dropna=False))
            for col in ZEEK_CATEGORICAL_COLS if col in combined.columns
        },
    }
    return report


def build_feature_info(combined: pd.DataFrame) -> dict:
    numeric_cols = [c for c in (ZEEK_NUMERIC_COLS + ENGINEERED_COLS) if c in combined.columns]
    feature_info = {"numeric": {}, "categorical": {}}

    for col in numeric_cols:
        s = combined[col]
        feature_info["numeric"][col] = {
            "dtype": str(s.dtype),
            "mean": safe_float(s.mean()),
            "std": safe_float(s.std()),
            "min": safe_float(s.min()),
            "max": safe_float(s.max()),
            "n_null": safe_int(s.isna().sum()),
        }

    for col in ZEEK_CATEGORICAL_COLS:
        if col in combined.columns:
            feature_info["categorical"][col] = {
                str(k): int(v) for k, v in combined[col].value_counts(dropna=False).head(30).to_dict().items()
            }

    return feature_info


# File discovery

def find_conn_logs(data_dir: str):
    pattern = os.path.join(data_dir, "**", "bro", "conn.log.labeled")
    files = glob.glob(pattern, recursive=True)

    result = []
    for f in sorted(files):
        parts = Path(f).parts
        try:
            bro_idx = parts.index("bro")
            parent_1 = parts[bro_idx - 1] if bro_idx - 1 >= 0 else ""
            parent_2 = parts[bro_idx - 2] if bro_idx - 2 >= 0 else ""

            if parent_1.startswith("CTU-"):
                scenario = parent_1
            elif parent_2.startswith("CTU-"):
                scenario = parent_2
            else:
                scenario = Path(f).parent.parent.name

        except (ValueError, IndexError):
            scenario = Path(f).parent.parent.name

        result.append((scenario, f))

    return result


# Chunked scenario processing

def process_scenario_to_chunks(
    scenario_name: str,
    filepath: str,
    chunks_dir: Path,
    sample_frac: float,
    seed: int,
    chunk_rows: int,
) -> List[Path]:
    """
    Parse one scenario in chunks and save chunk parquet files.
    Returns list of chunk parquet paths.
    """
    chunk_paths: List[Path] = []
    chunk_idx = 0
    total_rows_saved = 0

    for chunk_df, line_no, malformed_rows in iter_conn_log_chunks(
        filepath=filepath,
        scenario_name=scenario_name,
        chunk_rows=chunk_rows,
    ):
        if chunk_df.empty:
            continue

        chunk_df = clean_and_engineer(chunk_df)
        chunk_df = assign_labels(chunk_df)

        if sample_frac < 1.0:
            before_n = len(chunk_df)
            chunk_df = stratified_sample_df(chunk_df, frac=sample_frac, seed=seed)
            log.info(
                "Sampled chunk for %s: %d -> %d rows",
                scenario_name,
                before_n,
                len(chunk_df),
            )

        if chunk_df.empty:
            del chunk_df
            gc.collect()
            continue

        chunk_df = chunk_df.sort_values("ts", na_position="last").reset_index(drop=True)
        chunk_df["split"] = get_split(scenario_name)
        validate_per_scenario_df(chunk_df, scenario_name)

        available_cols = [c for c in FINAL_FEATURE_COLS if c in chunk_df.columns]
        chunk_df = chunk_df[available_cols]

        chunk_path = chunks_dir / f"{scenario_name}_part{chunk_idx:04d}.parquet"
        chunk_df.to_parquet(chunk_path, index=False)
        chunk_paths.append(chunk_path)

        total_rows_saved += len(chunk_df)
        log.info(
            "%s: saved chunk %s (%d rows, total saved=%d)",
            scenario_name,
            chunk_path.name,
            len(chunk_df),
            total_rows_saved,
        )

        chunk_idx += 1
        del chunk_df
        gc.collect()

    if not chunk_paths:
        raise AssertionError(f"{scenario_name}: no chunk parquet files were produced.")

    log.info("%s: completed chunk processing with %d chunk files", scenario_name, len(chunk_paths))
    return chunk_paths


def combine_scenario_chunks_to_final(
    scenario_name: str,
    chunk_paths: List[Path],
    scenario_out_dir: Path,
) -> Path:
    """
    Combine per-scenario chunk parquet files into one final scenario parquet.
    """
    parts = []
    for p in sorted(chunk_paths):
        part = pd.read_parquet(p)
        parts.append(part)

    scenario_df = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()

    scenario_path = scenario_out_dir / f"{scenario_name}.parquet"
    scenario_df.to_parquet(scenario_path, index=False)

    log.info("Saved final scenario parquet: %s (%d rows)", scenario_path.name, len(scenario_df))

    del scenario_df
    gc.collect()

    return scenario_path


# Main

def main():
    parser = argparse.ArgumentParser(description="IoT-23 preprocessing pipeline with chunked processing")
    parser.add_argument("--data_dir", required=True, help="Root directory of IoT-23 dataset")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--sample", type=float, default=1.0, help="Fraction of rows to keep per scenario")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--chunk_rows", type=int, default=500_000, help="Rows per parsed chunk before writing parquet")
    parser.add_argument(
        "--strict_benign_train",
        action="store_true",
        help="Fail if the train split contains zero benign rows"
    )
    args = parser.parse_args()

    if not (0 < args.sample <= 1.0):
        raise ValueError("--sample must be in the interval (0, 1].")

    validate_static_split_config()

    out_path = Path(args.out_dir) / "iot23"
    out_path.mkdir(parents=True, exist_ok=True)

    chunks_dir = out_path / "processed_chunks"
    scenario_out_dir = out_path / "processed_scenarios"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    scenario_out_dir.mkdir(parents=True, exist_ok=True)

    log_files = find_conn_logs(args.data_dir)
    if not log_files:
        raise FileNotFoundError(f"No conn.log.labeled files found under: {args.data_dir}")

    discovered_scenarios = [scenario for scenario, _ in log_files]
    if len(discovered_scenarios) != len(set(discovered_scenarios)):
        raise AssertionError("Duplicate scenario names found among discovered files.")

    log.info("Found %d scenario files", len(log_files))

    # Pass 1: chunk-process each scenario and combine chunk files to final

    scenario_paths: List[Path] = []

    for scenario_name, filepath in log_files:
        log.info("Parsing %s ...", scenario_name)

        chunk_paths = process_scenario_to_chunks(
            scenario_name=scenario_name,
            filepath=filepath,
            chunks_dir=chunks_dir,
            sample_frac=args.sample,
            seed=args.seed,
            chunk_rows=args.chunk_rows,
        )

        scenario_path = combine_scenario_chunks_to_final(
            scenario_name=scenario_name,
            chunk_paths=chunk_paths,
            scenario_out_dir=scenario_out_dir,
        )
        scenario_paths.append(scenario_path)

        for p in chunk_paths:
            try:
                p.unlink()
            except OSError:
                log.warning("Could not delete chunk file %s", p)

        gc.collect()

    if not scenario_paths:
        raise AssertionError("No final scenario parquet files were produced.")

    # Pass 2: combine final scenario parquet files

    log.info("Combining final scenario parquet files ...")
    combined_parts = []
    label_stats_frames = []

    for p in sorted(scenario_paths):
        log.info("Loading %s for final combine", p.name)
        part = pd.read_parquet(p)
        combined_parts.append(part)

        stats = (
            part.groupby(["scenario", "split", "label", "detailed_label", "label_binary", "label_phase"])
            .size()
            .reset_index(name="count")
        )
        label_stats_frames.append(stats)

    combined = pd.concat(combined_parts, ignore_index=True)
    del combined_parts
    gc.collect()

    combined = combined[[c for c in FINAL_FEATURE_COLS if c in combined.columns]]

    validate_combined_df(
        combined=combined,
        discovered_scenarios=discovered_scenarios,
        strict_benign_train=args.strict_benign_train,
    )

    log.info("Total flows parsed: %d", len(combined))
    log.info("Scenarios parsed: %d", combined["scenario"].nunique())
    log.info("Label distribution:\n%s", combined["label_binary"].value_counts(dropna=False).to_string())
    log.info("Phase distribution:\n%s", combined["label_phase"].value_counts(dropna=False).to_string())

    benign_by_split = combined.groupby("split")["label_binary"].apply(lambda s: int((s == 0).sum()))
    malicious_by_split = combined.groupby("split")["label_binary"].apply(lambda s: int((s == 1).sum()))
    log.info("Benign by split:\n%s", benign_by_split.to_string())
    log.info("Malicious by split:\n%s", malicious_by_split.to_string())

    # save final datasets
    combined.to_parquet(out_path / "all_flows.parquet", index=False)
    log.info("Saved all_flows.parquet")

    for split_name in ["train", "val", "test"]:
        subset = combined[combined["split"] == split_name]
        subset.to_parquet(out_path / f"{split_name}.parquet", index=False)
        log.info("Saved %s.parquet (%d rows)", split_name, len(subset))

    # label stats
    label_df = pd.concat(label_stats_frames, ignore_index=True)
    label_df.to_csv(out_path / "label_stats.csv", index=False)
    log.info("Saved label_stats.csv")

    # audit outputs
    scenario_summary = build_scenario_summary(combined)
    scenario_summary.to_csv(out_path / "scenario_summary.csv", index=False)
    log.info("Saved scenario_summary.csv")

    label_mapping_report = build_label_mapping_report(combined)
    label_mapping_report.to_csv(out_path / "label_mapping_report.csv", index=False)
    log.info("Saved label_mapping_report.csv")

    split_summary = build_split_summary(combined)
    save_json(split_summary, out_path / "split_summary.json")
    log.info("Saved split_summary.json")

    data_quality_report = build_data_quality_report(combined)
    save_json(data_quality_report, out_path / "data_quality_report.json")
    log.info("Saved data_quality_report.json")

    feature_info = build_feature_info(combined)
    save_json(feature_info, out_path / "feature_info.json")
    log.info("Saved feature_info.json")

    print("\n── Quick summary ──────────────────────────────────")
    print(f"  Scenarios parsed : {combined['scenario'].nunique()}")
    print(f"  Total flows      : {len(combined):,}")
    print(f"  Train flows      : {(combined['split'] == 'train').sum():,}")
    print(f"  Val flows        : {(combined['split'] == 'val').sum():,}")
    print(f"  Test flows       : {(combined['split'] == 'test').sum():,}")
    print(f"  Benign           : {(combined['label_binary'] == 0).sum():,}")
    print(f"  Malicious        : {(combined['label_binary'] == 1).sum():,}")
    print("───────────────────────────────────────────────────\n")

    log.info("Done. Output in: %s", out_path)


if __name__ == "__main__":
    main()
