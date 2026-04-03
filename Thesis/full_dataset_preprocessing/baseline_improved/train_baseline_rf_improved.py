from __future__ import annotations

import argparse
import gc
import logging
from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from baseline_common import (
    CATEGORICAL_COLS,
    FEATURE_COLS,
    NUMERIC_COLS,
    count_rows,
    evaluate_split_in_batches,
    sample_training_split,
    save_json,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def build_pipeline(seed: int, n_estimators: int, n_jobs: int) -> Pipeline:
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
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=seed,
        n_jobs=n_jobs,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Memory-safer Random Forest baseline for large IoT-23 splits")
    parser.add_argument("--data_dir", required=True, help="Directory containing train.parquet, val.parquet, test.parquet")
    parser.add_argument("--out_dir", required=True, help="Directory to save model and outputs")
    parser.add_argument("--target_col", default="label_binary")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=250_000)
    parser.add_argument("--train_sample_frac", type=float, default=0.10, help="Post-split stratified sampling fraction for train only")
    parser.add_argument("--max_train_rows", type=int, default=3_000_000, help="Optional hard cap after post-split sampling")
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--n_jobs", type=int, default=4)
    args = parser.parse_args()

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
    log.info("Sampled train label distribution:\n%s", train_df[args.target_col].value_counts(dropna=False).to_string())

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[args.target_col]

    pipeline = build_pipeline(seed=args.seed, n_estimators=args.n_estimators, n_jobs=args.n_jobs)
    log.info("Training Random Forest on sampled train split...")
    pipeline.fit(X_train, y_train)

    model_path = out_dir / "rf_model.joblib"
    joblib.dump(pipeline, model_path)
    log.info("Saved model to %s", model_path)

    del X_train, y_train, train_df
    gc.collect()

    log.info("Running full validation evaluation in batches...")
    val_metrics = evaluate_split_in_batches(
        pipeline=pipeline,
        split_path=val_path,
        split_name="val",
        out_path=out_dir / "rf_val_predictions.parquet",
        target_col=args.target_col,
        batch_size=args.batch_size,
    )

    log.info("Running full test evaluation in batches...")
    test_metrics = evaluate_split_in_batches(
        pipeline=pipeline,
        split_path=test_path,
        split_name="test",
        out_path=out_dir / "rf_test_predictions.parquet",
        target_col=args.target_col,
        batch_size=args.batch_size,
    )

    run_summary = {
        "model": "RandomForestClassifier",
        "target_col": args.target_col,
        "seed": int(args.seed),
        "batch_size": int(args.batch_size),
        "train_sample_frac": float(args.train_sample_frac),
        "max_train_rows": int(args.max_train_rows) if args.max_train_rows is not None else None,
        "n_estimators": int(args.n_estimators),
        "n_jobs": int(args.n_jobs),
        "input_rows": {
            "train_full": int(count_rows(train_path)),
            "val_full": int(count_rows(val_path)),
            "test_full": int(count_rows(test_path)),
        },
        "sampled_train_rows": sampled_train_rows,
        "features": {
            "numeric": NUMERIC_COLS,
            "categorical": CATEGORICAL_COLS,
        },
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "outputs": {
            "model_path": str(model_path),
            "val_predictions_path": str(out_dir / "rf_val_predictions.parquet"),
            "test_predictions_path": str(out_dir / "rf_test_predictions.parquet"),
        },
    }
    run_summary["notes"] = [
        "Training used post-split stratified sampling on train.parquet only.",
        "Validation and test evaluation were run over the full stored splits in batches.",
        "This script is designed to reduce peak RAM use compared with loading all splits fully into pandas at once.",
    ]
    summary_path = out_dir / "rf_run_summary.json"
    save_json(run_summary, summary_path)

    log.info("Validation metrics: %s", val_metrics)
    log.info("Test metrics: %s", test_metrics)
    log.info("Saved run summary to %s", summary_path)


if __name__ == "__main__":
    main()
