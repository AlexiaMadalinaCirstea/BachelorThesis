"""
This creates a small but representative IoT-23 sample for local testing

What it preserves:
- original folder structure
- all header lines
- label distribution as much as possible
- all scenarios

Outputs:
- sampled conn.log.labeled files
- manifest.json
- sample_summary.csv

Usage:
    python local_test.py \
        --data_dir "/path/to/iot23" \
        --out_dir "/path/to/iot23_sample" \
        --n_rows 2000 \
        --seed 42
"""

import os
import re
import glob
import json
import argparse
import logging
import random
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def extract_label_from_line(line: str) -> str:
    """
    Extract label from IoT-23 conn.log.labeled data line.
    """
    if line.startswith("#"):
        return None

    parts = line.split("\t")
    if len(parts) < 21:
        return "unknown"

    tail = parts[20]
    tail_parts = [p.strip() for p in re.split(r"\s{3,}", tail.strip()) if p.strip()]

    if len(tail_parts) >= 2:
        return tail_parts[1].lower().strip()

    return "unknown"


def stratified_sample_lines(data_lines: list, n: int, seed: int = 42) -> list:
    """
    Sample approximately n data lines while preserving label distribution.
    Ensures at least one sample per available label when possible.
    """
    if n >= len(data_lines):
        return list(data_lines)

    rng = random.Random(seed)
    label_to_indices = defaultdict(list)

    for i, line in enumerate(data_lines):
        label = extract_label_from_line(line)
        if label is not None:
            label_to_indices[label].append(i)

    total = len(data_lines)
    sampled_indices = set()

    # First pass: proportional allocation
    provisional = {}
    for label, indices in label_to_indices.items():
        proportion = len(indices) / total
        k = max(1, round(proportion * n))
        k = min(k, len(indices))
        provisional[label] = k

    # Fix overshoot
    total_requested = sum(provisional.values())
    if total_requested > n:
        labels_sorted = sorted(
            provisional.keys(),
            key=lambda x: provisional[x],
            reverse=True
        )
        i = 0
        while total_requested > n and i < len(labels_sorted):
            label = labels_sorted[i]
            if provisional[label] > 1:
                provisional[label] -= 1
                total_requested -= 1
            else:
                i += 1

    # Draw samples
    for label, indices in label_to_indices.items():
        k = provisional[label]
        chosen = rng.sample(indices, k)
        sampled_indices.update(chosen)

    # If undershot, top up from remaining rows
    if len(sampled_indices) < n:
        remaining = [i for i in range(len(data_lines)) if i not in sampled_indices]
        extra = rng.sample(remaining, min(n - len(sampled_indices), len(remaining)))
        sampled_indices.update(extra)

    sampled_list = sorted(sampled_indices)
    return [data_lines[i] for i in sampled_list]


def find_conn_logs(data_dir: str):
    pattern = os.path.join(data_dir, "**", "bro", "conn.log.labeled")
    files = glob.glob(pattern, recursive=True)
    result = []

    for src in sorted(files):
        rel = os.path.relpath(src, data_dir)
        parts = Path(src).parts

        try:
            bro_idx = parts.index("bro")
            parent_1 = parts[bro_idx - 1] if bro_idx - 1 >= 0 else ""
            parent_2 = parts[bro_idx - 2] if bro_idx - 2 >= 0 else ""

            if parent_1.startswith("CTU-"):
                scenario = parent_1
            elif parent_2.startswith("CTU-"):
                scenario = parent_2
            else:
                scenario = Path(src).parent.parent.name

        except (ValueError, IndexError):
            scenario = Path(src).parent.parent.name

        result.append((scenario, src, rel))

    return result


def sample_conn_log(src_path: str, dst_path: str, n_rows: int, seed: int) -> dict:
    header_lines = []
    data_lines = []

    with open(src_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if line.startswith("#"):
                header_lines.append(line)
            else:
                data_lines.append(line)

    original_n = len(data_lines)
    sampled_lines = stratified_sample_lines(data_lines, n_rows, seed=seed)

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, "w", encoding="utf-8") as f:
        for line in header_lines:
            f.write(line + "\n")
        for line in sampled_lines:
            f.write(line + "\n")

    # Summaries
    orig_label_counts = defaultdict(int)
    samp_label_counts = defaultdict(int)

    for line in data_lines:
        label = extract_label_from_line(line)
        if label:
            orig_label_counts[label] += 1

    for line in sampled_lines:
        label = extract_label_from_line(line)
        if label:
            samp_label_counts[label] += 1

    labels_lost = sorted(set(orig_label_counts.keys()) - set(samp_label_counts.keys()))

    return {
        "scenario": Path(src_path).parent.parent.name,
        "src_path": src_path,
        "dst_path": dst_path,
        "original_rows": int(original_n),
        "sampled_rows": int(len(sampled_lines)),
        "n_header_lines": int(len(header_lines)),
        "original_label_counts": dict(sorted(orig_label_counts.items())),
        "sampled_label_counts": dict(sorted(samp_label_counts.items())),
        "labels_lost_in_sample": labels_lost,
    }


def main():
    parser = argparse.ArgumentParser(description="Create a representative local IoT-23 sample")
    parser.add_argument("--data_dir", required=True, help="Root directory of IoT-23 dataset")
    parser.add_argument("--out_dir", required=True, help="Output directory for sampled dataset")
    parser.add_argument("--n_rows", type=int, default=2000, help="Rows to sample per scenario")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    manifest_path = out_dir / "manifest.json"

    if manifest_path.exists() and not args.force:
        raise FileExistsError(
            f"{manifest_path} already exists. Use --force if you want to overwrite the sample."
        )

    log_files = find_conn_logs(args.data_dir)
    if not log_files:
        raise FileNotFoundError(f"No conn.log.labeled files found under {args.data_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for scenario, src_path, rel_path in log_files:
        dst_path = out_dir / rel_path
        summary = sample_conn_log(src_path, str(dst_path), n_rows=args.n_rows, seed=args.seed)
        summaries.append(summary)

        log.info(
            "%s | original=%d sampled=%d lost_labels=%s",
            scenario,
            summary["original_rows"],
            summary["sampled_rows"],
            summary["labels_lost_in_sample"],
        )

    # Save summary CSV
    summary_rows = []
    for s in summaries:
        summary_rows.append({
            "scenario": s["scenario"],
            "original_rows": s["original_rows"],
            "sampled_rows": s["sampled_rows"],
            "n_header_lines": s["n_header_lines"],
            "labels_lost_in_sample": ",".join(s["labels_lost_in_sample"]),
            "n_unique_labels_original": len(s["original_label_counts"]),
            "n_unique_labels_sampled": len(s["sampled_label_counts"]),
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("scenario").reset_index(drop=True)
    summary_df.to_csv(out_dir / "sample_summary.csv", index=False)

    manifest = {
        "created_at": datetime.now().isoformat(),
        "data_dir": str(args.data_dir),
        "out_dir": str(args.out_dir),
        "n_rows": int(args.n_rows),
        "seed": int(args.seed),
        "n_scenarios": int(len(log_files)),
        "sample_summary_csv": str(out_dir / "sample_summary.csv"),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    log.info("Done.")
    log.info("Sample written to: %s", out_dir)
    log.info("Manifest: %s", manifest_path)
    log.info("Summary: %s", out_dir / "sample_summary.csv")


if __name__ == "__main__":
    main()