from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


DEFAULT_THRESHOLDS = [
    0.50,
    0.30,
    0.20,
    0.15,
    0.12,
    0.10,
    0.08,
    0.05,
    0.02,
    0.01,
]


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den != 0 else 0.0


def f1_score_from_pr(precision: float, recall: float) -> float:
    den = precision + recall
    return float(2 * precision * recall / den) if den != 0 else 0.0


def metrics_from_confusion(cm: np.ndarray) -> dict:
    tn, fp = int(cm[0, 0]), int(cm[0, 1])
    fn, tp = int(cm[1, 0]), int(cm[1, 1])

    support_0 = tn + fp
    support_1 = fn + tp
    total = support_0 + support_1

    precision_0 = safe_div(tn, tn + fn)
    recall_0 = safe_div(tn, tn + fp)
    f1_0 = f1_score_from_pr(precision_0, recall_0)

    precision_1 = safe_div(tp, tp + fp)
    recall_1 = safe_div(tp, tp + fn)
    f1_1 = f1_score_from_pr(precision_1, recall_1)

    accuracy = safe_div(tn + tp, total)
    balanced_accuracy = (recall_0 + recall_1) / 2.0

    precision_macro = (precision_0 + precision_1) / 2.0
    recall_macro = (recall_0 + recall_1) / 2.0
    f1_macro = (f1_0 + f1_1) / 2.0

    precision_weighted = safe_div(
        precision_0 * support_0 + precision_1 * support_1,
        total,
    )
    recall_weighted = safe_div(
        recall_0 * support_0 + recall_1 * support_1,
        total,
    )
    f1_weighted = safe_div(
        f1_0 * support_0 + f1_1 * support_1,
        total,
    )

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "precision_benign": precision_0,
        "recall_benign": recall_0,
        "f1_benign": f1_0,
        "support_benign": support_0,
        "precision_attack": precision_1,
        "recall_attack": recall_1,
        "f1_attack": f1_1,
        "support_attack": support_1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "n_rows": total,
    }


def parse_thresholds(raw: str | None) -> list[float]:
    if not raw:
        return DEFAULT_THRESHOLDS
    vals = []
    for x in raw.split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(float(x))
    vals = sorted(set(vals), reverse=True)
    return vals


def iter_prediction_batches(parquet_path: Path, batch_size: int) -> Iterable[pd.DataFrame]:
    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(batch_size=batch_size, columns=["label_binary", "y_score"]):
        yield batch.to_pandas()


def evaluate_thresholds_for_file(
    parquet_path: Path,
    thresholds: list[float],
    batch_size: int,
) -> pd.DataFrame:
    cms = {thr: np.zeros((2, 2), dtype=np.int64) for thr in thresholds}

    chunk_count = 0
    for df in iter_prediction_batches(parquet_path, batch_size=batch_size):
        chunk_count += 1
        y_true = df["label_binary"].to_numpy(dtype=np.int8, copy=False)
        y_score = df["y_score"].to_numpy(dtype=np.float32, copy=False)

        for thr in thresholds:
            y_pred = (y_score >= thr).astype(np.int8)
            cms[thr][0, 0] += int(((y_true == 0) & (y_pred == 0)).sum())
            cms[thr][0, 1] += int(((y_true == 0) & (y_pred == 1)).sum())
            cms[thr][1, 0] += int(((y_true == 1) & (y_pred == 0)).sum())
            cms[thr][1, 1] += int(((y_true == 1) & (y_pred == 1)).sum())

        if chunk_count % 100 == 0:
            log.info("%s: processed %d chunks", parquet_path.name, chunk_count)

    rows = []
    for thr in thresholds:
        row = {"threshold": thr}
        row.update(metrics_from_confusion(cms[thr]))
        rows.append(row)

    return pd.DataFrame(rows).sort_values("threshold", ascending=False).reset_index(drop=True)


def load_run_summary(run_dir: Path) -> dict:
    path = run_dir / "xgb_run_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing run summary: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_line_plot(
    df: pd.DataFrame,
    x_col: str,
    y_cols: list[str],
    out_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    plt.figure(figsize=(10, 6))
    for col in y_cols:
        plt.plot(df[x_col], df[col], marker="o", label=col)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def make_score_histogram(
    parquet_path: Path,
    out_path: Path,
    title: str,
    batch_size: int,
    max_points_per_class: int = 300_000,
) -> None:
    benign_scores = []
    attack_scores = []
    benign_kept = 0
    attack_kept = 0

    for df in iter_prediction_batches(parquet_path, batch_size=batch_size):
        y_true = df["label_binary"].to_numpy(dtype=np.int8, copy=False)
        y_score = df["y_score"].to_numpy(dtype=np.float32, copy=False)

        benign_mask = y_true == 0
        attack_mask = y_true == 1

        benign_chunk = y_score[benign_mask]
        attack_chunk = y_score[attack_mask]

        if benign_kept < max_points_per_class and len(benign_chunk) > 0:
            take = min(max_points_per_class - benign_kept, len(benign_chunk))
            benign_scores.append(benign_chunk[:take])
            benign_kept += take

        if attack_kept < max_points_per_class and len(attack_chunk) > 0:
            take = min(max_points_per_class - attack_kept, len(attack_chunk))
            attack_scores.append(attack_chunk[:take])
            attack_kept += take

        if benign_kept >= max_points_per_class and attack_kept >= max_points_per_class:
            break

    benign_scores = np.concatenate(benign_scores) if benign_scores else np.array([], dtype=np.float32)
    attack_scores = np.concatenate(attack_scores) if attack_scores else np.array([], dtype=np.float32)

    plt.figure(figsize=(10, 6))
    if len(benign_scores) > 0:
        plt.hist(benign_scores, bins=60, alpha=0.6, density=True, label="benign")
    if len(attack_scores) > 0:
        plt.hist(attack_scores, bins=60, alpha=0.6, density=True, label="attack")
    plt.xlabel("Predicted attack probability (y_score)")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def summarize_best_thresholds(df: pd.DataFrame) -> dict:
    best_macro_f1 = df.loc[df["f1_macro"].idxmax()].to_dict()
    best_attack_f1 = df.loc[df["f1_attack"].idxmax()].to_dict()
    best_balanced_acc = df.loc[df["balanced_accuracy"].idxmax()].to_dict()
    best_attack_recall = df.loc[df["recall_attack"].idxmax()].to_dict()
    return {
        "best_macro_f1": best_macro_f1,
        "best_attack_f1": best_attack_f1,
        "best_balanced_accuracy": best_balanced_acc,
        "best_attack_recall": best_attack_recall,
    }


def analyze_one_run(run_dir: Path, thresholds: list[float], batch_size: int) -> None:
    run_name = run_dir.name
    summary = load_run_summary(run_dir)

    val_path = run_dir / "xgb_val_predictions.parquet"
    test_path = run_dir / "xgb_test_predictions.parquet"

    if not val_path.exists():
        raise FileNotFoundError(f"Missing validation predictions: {val_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test predictions: {test_path}")

    out_dir = run_dir / "threshold_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Analyzing %s", run_name)

    val_df = evaluate_thresholds_for_file(val_path, thresholds, batch_size)
    val_df.insert(0, "split", "val")
    test_df = evaluate_thresholds_for_file(test_path, thresholds, batch_size)
    test_df.insert(0, "split", "test")

    combined = pd.concat([val_df, test_df], ignore_index=True)
    combined.to_csv(out_dir / "threshold_metrics.csv", index=False)

    val_df.to_csv(out_dir / "val_threshold_metrics.csv", index=False)
    test_df.to_csv(out_dir / "test_threshold_metrics.csv", index=False)

    make_line_plot(
        val_df,
        x_col="threshold",
        y_cols=["f1_macro", "balanced_accuracy", "recall_attack", "precision_attack"],
        out_path=out_dir / "val_threshold_curves.png",
        title=f"{run_name} - Validation threshold analysis",
        xlabel="Decision threshold",
        ylabel="Metric value",
    )

    make_line_plot(
        test_df,
        x_col="threshold",
        y_cols=["f1_macro", "balanced_accuracy", "recall_attack", "precision_attack"],
        out_path=out_dir / "test_threshold_curves.png",
        title=f"{run_name} - Test threshold analysis",
        xlabel="Decision threshold",
        ylabel="Metric value",
    )

    make_line_plot(
        test_df,
        x_col="threshold",
        y_cols=["recall_benign", "recall_attack"],
        out_path=out_dir / "test_class_recalls.png",
        title=f"{run_name} - Test class recalls by threshold",
        xlabel="Decision threshold",
        ylabel="Recall",
    )

    make_score_histogram(
        parquet_path=val_path,
        out_path=out_dir / "val_score_histogram.png",
        title=f"{run_name} - Validation score distribution",
        batch_size=batch_size,
    )

    make_score_histogram(
        parquet_path=test_path,
        out_path=out_dir / "test_score_histogram.png",
        title=f"{run_name} - Test score distribution",
        batch_size=batch_size,
    )

    summary_out = {
        "run_name": run_name,
        "original_run_summary_threshold": summary.get("decision_threshold"),
        "thresholds_evaluated": thresholds,
        "validation_best": summarize_best_thresholds(val_df),
        "test_best": summarize_best_thresholds(test_df),
    }

    save_path = out_dir / "threshold_analysis_summary.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(summary_out, f, indent=2)

    readme_text = f"""Threshold analysis for {run_name}.

This analysis was performed offline using the saved prediction files:
- xgb_val_predictions.parquet
- xgb_test_predictions.parquet

No retraining was performed. Metrics were recomputed for multiple decision thresholds
using the stored predicted probabilities (y_score). This analysis is intended to assess
threshold sensitivity, class-specific trade-offs, and whether threshold adjustment alone
can resolve the mismatch between validation and test behavior.

Outputs:
- threshold_metrics.csv
- val_threshold_metrics.csv
- test_threshold_metrics.csv
- val_threshold_curves.png
- test_threshold_curves.png
- test_class_recalls.png
- val_score_histogram.png
- test_score_histogram.png
- threshold_analysis_summary.json
"""
    (out_dir / "README_THRESHOLD_ANALYSIS.txt").write_text(readme_text, encoding="utf-8")

    log.info("Saved threshold analysis outputs to %s", out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline threshold analysis for saved non-LOSO XGBoost prediction files")
    parser.add_argument(
        "--base_dir",
        required=True,
        help="Directory containing xgb_baseline_improved_v*/ folders",
    )
    parser.add_argument(
        "--run_dirs",
        nargs="*",
        default=None,
        help="Optional specific run folder names inside base_dir. If omitted, all xgb_baseline_improved_v* folders are used.",
    )
    parser.add_argument(
        "--thresholds",
        default=None,
        help="Comma-separated thresholds, e.g. 0.5,0.3,0.2,0.15,0.1,0.05",
    )
    parser.add_argument("--batch_size", type=int, default=200_000)
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    thresholds = parse_thresholds(args.thresholds)

    if args.run_dirs:
        run_dirs = [base_dir / name for name in args.run_dirs]
    else:
        run_dirs = sorted(
            [p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith("xgb_baseline_improved_v")]
        )

    if not run_dirs:
        raise FileNotFoundError("No matching run directories found.")

    log.info("Thresholds to evaluate: %s", thresholds)
    log.info("Found %d run directories", len(run_dirs))

    for run_dir in run_dirs:
        analyze_one_run(run_dir, thresholds, args.batch_size)

    log.info("Done.")


if __name__ == "__main__":
    main()