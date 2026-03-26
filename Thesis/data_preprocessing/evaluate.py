"""
I built this as a reusable evaluation script for thesis experiments.

What it supports:
- binary or multiclass evaluation
- overall metrics
- per-class metrics
- per-scenario metrics
- confusion matrix
- bootstrap confidence intervals
- JSON + CSV outputs

Typical usage:
    python evaluate.py \
        --pred_file /path/to/predictions.parquet \
        --out_dir /path/to/eval/binary_run1 \
        --task binary \
        --y_true_col label_binary \
        --y_pred_col y_pred \
        --y_score_col y_score

    python evaluate.py \
        --pred_file /path/to/predictions.parquet \
        --out_dir /path/to/eval/phase_run1 \
        --task multiclass \
        --y_true_col label_phase \
        --y_pred_col y_pred_phase
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)



# Helpers


def save_json(obj: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def safe_float(x) -> Optional[float]:
    if x is None:
        return None
    if pd.isna(x):
        return None
    return float(x)


def load_table(path: str) -> pd.DataFrame:
    path = str(path)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError("pred_file must be .parquet or .csv")


def bootstrap_metric(
    y_true,
    y_pred,
    metric_fn,
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values = []

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        try:
            values.append(metric_fn(yt, yp))
        except Exception:
            continue

    if not values:
        return {"mean": None, "ci_low": None, "ci_high": None}

    arr = np.array(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "ci_low": float(np.quantile(arr, 0.025)),
        "ci_high": float(np.quantile(arr, 0.975)),
    }


def validate_inputs(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    y_score_col: Optional[str],
) -> None:
    required = {"scenario", y_true_col, y_pred_col}
    missing = required - set(df.columns)
    if missing:
        raise AssertionError(f"Missing required columns: {sorted(missing)}")

    if y_score_col and y_score_col not in df.columns:
        raise AssertionError(f"Missing y_score_col: {y_score_col}")

    if df.empty:
        raise AssertionError("Prediction file is empty.")

    n_missing_true = int(df[y_true_col].isna().sum())
    n_missing_pred = int(df[y_pred_col].isna().sum())
    if n_missing_true > 0 or n_missing_pred > 0:
        raise AssertionError(
            f"Found missing labels/predictions: y_true missing={n_missing_true}, y_pred missing={n_missing_pred}"
        )



# Metric computation


def compute_overall_metrics_binary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, object]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )

    out = {
        "n_samples": int(len(y_true)),
        "accuracy": safe_float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": safe_float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": safe_float(precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)[0]),
        "recall_macro": safe_float(precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)[1]),
        "f1_macro": safe_float(precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)[2]),
        "precision_weighted": safe_float(precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)[0]),
        "recall_weighted": safe_float(precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)[1]),
        "f1_weighted": safe_float(precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)[2]),
        "per_class": {
            "0": {
                "precision": safe_float(precision[0]),
                "recall": safe_float(recall[0]),
                "f1": safe_float(f1[0]),
                "support": int(support[0]),
            },
            "1": {
                "precision": safe_float(precision[1]),
                "recall": safe_float(recall[1]),
                "f1": safe_float(f1[1]),
                "support": int(support[1]),
            },
        },
        "bootstrap_ci": {
            "accuracy": bootstrap_metric(y_true, y_pred, accuracy_score, n_boot=n_boot, seed=seed),
            "f1_macro": bootstrap_metric(
                y_true,
                y_pred,
                lambda yt, yp: precision_recall_fscore_support(yt, yp, average="macro", zero_division=0)[2],
                n_boot=n_boot,
                seed=seed + 1,
            ),
            "f1_weighted": bootstrap_metric(
                y_true,
                y_pred,
                lambda yt, yp: precision_recall_fscore_support(yt, yp, average="weighted", zero_division=0)[2],
                n_boot=n_boot,
                seed=seed + 2,
            ),
            "recall_attack": bootstrap_metric(
                y_true,
                y_pred,
                lambda yt, yp: precision_recall_fscore_support(yt, yp, labels=[0, 1], zero_division=0)[1][1],
                n_boot=n_boot,
                seed=seed + 3,
            ),
        },
    }

    if y_score is not None:
        try:
            out["roc_auc"] = safe_float(roc_auc_score(y_true, y_score))
        except Exception:
            out["roc_auc"] = None
        try:
            out["pr_auc"] = safe_float(average_precision_score(y_true, y_score))
        except Exception:
            out["pr_auc"] = None

    return out


def compute_overall_metrics_multiclass(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, object]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    per_class = {}
    for i, label in enumerate(labels):
        per_class[str(label)] = {
            "precision": safe_float(precision[i]),
            "recall": safe_float(recall[i]),
            "f1": safe_float(f1[i]),
            "support": int(support[i]),
        }

    out = {
        "n_samples": int(len(y_true)),
        "accuracy": safe_float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": safe_float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": safe_float(precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)[0]),
        "recall_macro": safe_float(precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)[1]),
        "f1_macro": safe_float(precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)[2]),
        "precision_weighted": safe_float(precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)[0]),
        "recall_weighted": safe_float(precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)[1]),
        "f1_weighted": safe_float(precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)[2]),
        "per_class": per_class,
        "bootstrap_ci": {
            "accuracy": bootstrap_metric(y_true, y_pred, accuracy_score, n_boot=n_boot, seed=seed),
            "f1_macro": bootstrap_metric(
                y_true,
                y_pred,
                lambda yt, yp: precision_recall_fscore_support(yt, yp, average="macro", zero_division=0)[2],
                n_boot=n_boot,
                seed=seed + 1,
            ),
            "f1_weighted": bootstrap_metric(
                y_true,
                y_pred,
                lambda yt, yp: precision_recall_fscore_support(yt, yp, average="weighted", zero_division=0)[2],
                n_boot=n_boot,
                seed=seed + 2,
            ),
        },
    }
    return out


def compute_per_scenario_metrics(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    task: str,
) -> pd.DataFrame:
    rows = []

    for scenario, g in df.groupby("scenario"):
        y_true = g[y_true_col].to_numpy()
        y_pred = g[y_pred_col].to_numpy()

        row = {
            "scenario": scenario,
            "n_rows": int(len(g)),
            "accuracy": safe_float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": safe_float(balanced_accuracy_score(y_true, y_pred)),
            "precision_macro": safe_float(precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)[0]),
            "recall_macro": safe_float(precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)[1]),
            "f1_macro": safe_float(precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)[2]),
            "precision_weighted": safe_float(precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)[0]),
            "recall_weighted": safe_float(precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)[1]),
            "f1_weighted": safe_float(precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)[2]),
        }

        if task == "binary":
            pr, rc, f1, sp = precision_recall_fscore_support(
                y_true, y_pred, labels=[0, 1], zero_division=0
            )
            row.update({
                "precision_benign": safe_float(pr[0]),
                "recall_benign": safe_float(rc[0]),
                "f1_benign": safe_float(f1[0]),
                "support_benign": int(sp[0]),
                "precision_attack": safe_float(pr[1]),
                "recall_attack": safe_float(rc[1]),
                "f1_attack": safe_float(f1[1]),
                "support_attack": int(sp[1]),
            })

        rows.append(row)

    return pd.DataFrame(rows).sort_values("scenario").reset_index(drop=True)



# Main


def main():
    parser = argparse.ArgumentParser(description="Evaluate predictions for IDS experiments")
    parser.add_argument("--pred_file", required=True, help="Parquet or CSV file with predictions")
    parser.add_argument("--out_dir", required=True, help="Output directory for evaluation reports")
    parser.add_argument("--task", choices=["binary", "multiclass"], required=True)
    parser.add_argument("--y_true_col", required=True)
    parser.add_argument("--y_pred_col", required=True)
    parser.add_argument("--y_score_col", default=None, help="Optional score/probability column for binary ROC-AUC / PR-AUC")
    parser.add_argument("--n_boot", type=int, default=1000, help="Bootstrap resamples for CIs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_table(args.pred_file)
    validate_inputs(df, args.y_true_col, args.y_pred_col, args.y_score_col)

    y_true = df[args.y_true_col].to_numpy()
    y_pred = df[args.y_pred_col].to_numpy()

    labels = sorted(pd.Series(y_true).astype(str).unique().tolist()) if args.task == "multiclass" else [0, 1]

    # Overall metrics
    if args.task == "binary":
        y_score = df[args.y_score_col].to_numpy() if args.y_score_col else None
        overall = compute_overall_metrics_binary(
            y_true=y_true,
            y_pred=y_pred,
            y_score=y_score,
            n_boot=args.n_boot,
            seed=args.seed,
        )
        cm_labels = [0, 1]
    else:
        overall = compute_overall_metrics_multiclass(
            y_true=y_true,
            y_pred=y_pred,
            labels=labels,
            n_boot=args.n_boot,
            seed=args.seed,
        )
        cm_labels = labels

    # Per-scenario metrics
    per_scenario = compute_per_scenario_metrics(
        df=df,
        y_true_col=args.y_true_col,
        y_pred_col=args.y_pred_col,
        task=args.task,
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=cm_labels)
    cm_df = pd.DataFrame(cm, index=[f"true_{x}" for x in cm_labels], columns=[f"pred_{x}" for x in cm_labels])

    # Classification report
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=cm_labels,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).T.reset_index().rename(columns={"index": "label"})

    # Save outputs
    save_json(overall, out_dir / "overall_metrics.json")
    per_scenario.to_csv(out_dir / "per_scenario_metrics.csv", index=False)
    cm_df.to_csv(out_dir / "confusion_matrix.csv", index=True)
    report_df.to_csv(out_dir / "classification_report.csv", index=False)

    run_meta = {
        "pred_file": str(args.pred_file),
        "task": args.task,
        "y_true_col": args.y_true_col,
        "y_pred_col": args.y_pred_col,
        "y_score_col": args.y_score_col,
        "n_rows": int(len(df)),
        "n_scenarios": int(df["scenario"].nunique()),
        "n_boot": int(args.n_boot),
        "seed": int(args.seed),
    }
    save_json(run_meta, out_dir / "run_meta.json")

    log.info("Saved evaluation outputs to %s", out_dir)
    log.info("Accuracy: %.4f", overall["accuracy"])
    log.info("F1 macro: %.4f", overall["f1_macro"])
    log.info("F1 weighted: %.4f", overall["f1_weighted"])


if __name__ == "__main__":
    main()