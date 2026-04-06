from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from baseline_common import (
    CATEGORICAL_COLS,
    FEATURE_COLS,
    NUMERIC_COLS,
    count_rows,
    sample_training_split,
    save_json,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def compute_scale_pos_weight(y: pd.Series) -> float:
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0:
        return 1.0
    return max(n_neg / n_pos, 1e-6)


def build_pipeline(
    seed: int,
    n_jobs: int,
    scale_pos_weight: float,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
) -> Pipeline:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_COLS),
        ("cat", categorical_transformer, CATEGORICAL_COLS),
    ])
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=n_jobs,
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_lambda=5.0,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den != 0 else 0.0


def _f1(precision: float, recall: float) -> float:
    den = precision + recall
    return float(2 * precision * recall / den) if den != 0 else 0.0


def _metrics_from_confusion(cm: np.ndarray, predictions_path: str, split_name: str) -> dict:
    # labels fixed as [0, 1]
    tn, fp = int(cm[0, 0]), int(cm[0, 1])
    fn, tp = int(cm[1, 0]), int(cm[1, 1])

    support_0 = tn + fp
    support_1 = fn + tp
    total = support_0 + support_1

    precision_0 = _safe_div(tn, tn + fn)
    recall_0 = _safe_div(tn, tn + fp)
    f1_0 = _f1(precision_0, recall_0)

    precision_1 = _safe_div(tp, tp + fp)
    recall_1 = _safe_div(tp, tp + fn)
    f1_1 = _f1(precision_1, recall_1)

    accuracy = _safe_div(tn + tp, total)

    precision_macro = (precision_0 + precision_1) / 2.0
    recall_macro = (recall_0 + recall_1) / 2.0
    f1_macro = (f1_0 + f1_1) / 2.0

    precision_weighted = _safe_div(
        precision_0 * support_0 + precision_1 * support_1,
        total,
    )
    recall_weighted = _safe_div(
        recall_0 * support_0 + recall_1 * support_1,
        total,
    )
    f1_weighted = _safe_div(
        f1_0 * support_0 + f1_1 * support_1,
        total,
    )

    return {
        "accuracy": accuracy,
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "confusion_matrix": {
            "labels": [0, 1],
            "matrix": cm.tolist(),
        },
        "per_class": {
            "0": {
                "precision": float(precision_0),
                "recall": float(recall_0),
                "f1": float(f1_0),
                "support": int(support_0),
            },
            "1": {
                "precision": float(precision_1),
                "recall": float(recall_1),
                "f1": float(f1_1),
                "support": int(support_1),
            },
        },
        "n_rows": int(total),
        "predictions_path": predictions_path,
        "split_name": split_name,
    }


def evaluate_split_in_batches_threshold(
    pipeline: Pipeline,
    split_path: Path,
    split_name: str,
    out_path: Path,
    target_col: str,
    batch_size: int,
    decision_threshold: float,
) -> dict:
    required_columns = ["scenario", target_col] + FEATURE_COLS
    parquet_file = pq.ParquetFile(split_path)

    writer = None
    cm = np.zeros((2, 2), dtype=np.int64)
    chunk_count = 0

    try:
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=required_columns):
            chunk_count += 1
            df = batch.to_pandas()
            y_true = df[target_col].to_numpy(dtype=np.int8, copy=False)

            X_batch = df[FEATURE_COLS]
            y_score = pipeline.predict_proba(X_batch)[:, 1]
            y_pred = (y_score >= decision_threshold).astype(np.int8)

            # update confusion matrix manually
            cm[0, 0] += int(((y_true == 0) & (y_pred == 0)).sum())
            cm[0, 1] += int(((y_true == 0) & (y_pred == 1)).sum())
            cm[1, 0] += int(((y_true == 1) & (y_pred == 0)).sum())
            cm[1, 1] += int(((y_true == 1) & (y_pred == 1)).sum())

            pred_df = pd.DataFrame({
                "scenario": df["scenario"].astype(str),
                "label_binary": y_true.astype(np.int8),
                "y_pred": y_pred.astype(np.int8),
                "y_score": y_score.astype(np.float32),
            })

            table = pa.Table.from_pandas(pred_df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema)
            writer.write_table(table)

            del df, X_batch, y_true, y_score, y_pred, pred_df, table, batch
            gc.collect()

        log.info("%s: processed %d prediction chunks", split_name, chunk_count)

    finally:
        if writer is not None:
            writer.close()

    return _metrics_from_confusion(
        cm=cm,
        predictions_path=str(out_path),
        split_name=split_name,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Memory-safer XGBoost baseline with configurable decision threshold")
    parser.add_argument("--data_dir", required=True, help="Directory containing train.parquet, val.parquet, test.parquet")
    parser.add_argument("--out_dir", required=True, help="Directory to save model and outputs")
    parser.add_argument("--target_col", default="label_binary")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=250_000)
    parser.add_argument("--train_sample_frac", type=float, default=0.10, help="Post-split stratified sampling fraction for train only")
    parser.add_argument("--max_train_rows", type=int, default=3_000_000, help="Optional hard cap after post-split sampling")
    parser.add_argument("--n_estimators", type=int, default=250)
    parser.add_argument("--max_depth", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--n_jobs", type=int, default=4)
    parser.add_argument("--decision_threshold", type=float, default=0.50, help="Probability threshold used to convert scores into class predictions")
    args = parser.parse_args()

    if not (0.0 < args.decision_threshold < 1.0):
        raise ValueError("--decision_threshold must be between 0 and 1")

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train.parquet"
    val_path = data_dir / "val.parquet"
    test_path = data_dir / "test.parquet"

    log.info("Sampling training split from %s", train_path)
    train_df = sample_training_split(
        path=train_path,
        target_col=args.target_col,
        batch_size=args.batch_size,
        sample_frac=args.train_sample_frac,
        seed=args.seed,
        max_rows=args.max_train_rows,
    )
    sampled_train_rows = int(len(train_df))
    log.info("Sampled train shape: %s", train_df.shape)
    log.info(
        "Sampled train label distribution:\n%s",
        train_df[args.target_col].value_counts(dropna=False).to_string()
    )

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[args.target_col]
    scale_pos_weight = compute_scale_pos_weight(y_train)

    pipeline = build_pipeline(
        seed=args.seed,
        n_jobs=args.n_jobs,
        scale_pos_weight=scale_pos_weight,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
    )

    log.info("Training XGBoost on sampled train split...")
    pipeline.fit(X_train, y_train)

    model_path = out_dir / "xgb_model.joblib"
    joblib.dump(pipeline, model_path)
    log.info("Saved model to %s", model_path)

    del X_train, y_train, train_df
    gc.collect()

    log.info("Running full validation evaluation in batches with threshold=%.4f...", args.decision_threshold)
    val_metrics = evaluate_split_in_batches_threshold(
        pipeline=pipeline,
        split_path=val_path,
        split_name="val",
        out_path=out_dir / "xgb_val_predictions.parquet",
        target_col=args.target_col,
        batch_size=args.batch_size,
        decision_threshold=args.decision_threshold,
    )

    log.info("Running full test evaluation in batches with threshold=%.4f...", args.decision_threshold)
    test_metrics = evaluate_split_in_batches_threshold(
        pipeline=pipeline,
        split_path=test_path,
        split_name="test",
        out_path=out_dir / "xgb_test_predictions.parquet",
        target_col=args.target_col,
        batch_size=args.batch_size,
        decision_threshold=args.decision_threshold,
    )

    run_summary = {
        "model": "XGBClassifier",
        "target_col": args.target_col,
        "seed": int(args.seed),
        "batch_size": int(args.batch_size),
        "train_sample_frac": float(args.train_sample_frac),
        "max_train_rows": int(args.max_train_rows) if args.max_train_rows is not None else None,
        "sampled_train_rows": sampled_train_rows,
        "n_estimators": int(args.n_estimators),
        "max_depth": int(args.max_depth),
        "learning_rate": float(args.learning_rate),
        "n_jobs": int(args.n_jobs),
        "decision_threshold": float(args.decision_threshold),
        "scale_pos_weight": float(scale_pos_weight),
        "input_rows": {
            "train_full": int(count_rows(train_path)),
            "val_full": int(count_rows(val_path)),
            "test_full": int(count_rows(test_path)),
        },
        "features": {
            "numeric": NUMERIC_COLS,
            "categorical": CATEGORICAL_COLS,
        },
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "outputs": {
            "model_path": str(model_path),
            "val_predictions_path": str(out_dir / "xgb_val_predictions.parquet"),
            "test_predictions_path": str(out_dir / "xgb_test_predictions.parquet"),
        },
        "notes": [
            "Training used post-split stratified sampling on train.parquet only.",
            "Validation and test evaluation were run over the full stored splits in batches.",
            "This version applies a manual decision threshold to predicted probabilities.",
            "This improved script avoids loading train, val, and test fully into pandas at the same time.",
        ],
    }
    save_json(run_summary, out_dir / "xgb_run_summary.json")

    log.info("Validation metrics: %s", val_metrics)
    log.info("Test metrics: %s", test_metrics)
    log.info("Saved run summary to %s", out_dir / "xgb_run_summary.json")


if __name__ == "__main__":
    main()