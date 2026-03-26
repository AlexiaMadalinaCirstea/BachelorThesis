"""
this is my first baseline
Here I train a Random Forest baseline for binary intrusion detection on processed IoT-23 data

Inputs:
    train.parquet
    val.parquet
    test.parquet

Outputs:
    rf_model.joblib
    rf_test_predictions.parquet
    rf_run_summary.json

Usage:
    python train_baseline.py \
        --data_dir "..\\Datasets\\processed_test\\iot23" \
        --out_dir "..\\Datasets\\processed_test\\iot23\\rf_baseline" \
        --target_col label_binary \
        --seed 42
"""

import json
import argparse
import logging
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

NUMERIC_COLS = [
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
]

CATEGORICAL_COLS = [
    "proto",
    "service",
    "conn_state",
]

META_COLS = [
    "scenario",
    "split",
    "label",
    "detailed_label",
    "label_phase",
    "ts",
]


def save_json(obj: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def load_split(data_dir: Path, split_name: str) -> pd.DataFrame:
    path = data_dir / f"{split_name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    df = pd.read_parquet(path)
    if df.empty:
        raise AssertionError(f"{split_name}.parquet is empty")
    return df


def validate_columns(df: pd.DataFrame, target_col: str) -> None:
    required = set(NUMERIC_COLS + CATEGORICAL_COLS + ["scenario", target_col])
    missing = required - set(df.columns)
    if missing:
        raise AssertionError(f"Missing required columns: {sorted(missing)}")


def build_pipeline(seed: int) -> Pipeline:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_COLS),
            ("cat", categorical_transformer, CATEGORICAL_COLS),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])
    return pipeline


def compute_basic_metrics(y_true, y_pred) -> dict:
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
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
    }


def main():
    parser = argparse.ArgumentParser(description="Train Random Forest baseline on processed IoT-23 data")
    parser.add_argument("--data_dir", required=True, help="Directory containing train/val/test parquet files")
    parser.add_argument("--out_dir", required=True, help="Directory to save model and predictions")
    parser.add_argument("--target_col", default="label_binary", help="Target column")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading splits from %s", data_dir)
    train_df = load_split(data_dir, "train")
    val_df = load_split(data_dir, "val")
    test_df = load_split(data_dir, "test")

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        validate_columns(df, args.target_col)
        log.info("%s shape: %s", name, df.shape)
        log.info("%s label distribution:\n%s", name, df[args.target_col].value_counts().to_string())

    X_train = train_df[NUMERIC_COLS + CATEGORICAL_COLS].copy()
    y_train = train_df[args.target_col].copy()

    X_val = val_df[NUMERIC_COLS + CATEGORICAL_COLS].copy()
    y_val = val_df[args.target_col].copy()

    X_test = test_df[NUMERIC_COLS + CATEGORICAL_COLS].copy()
    y_test = test_df[args.target_col].copy()

    pipeline = build_pipeline(seed=args.seed)

    log.info("Training Random Forest...")
    pipeline.fit(X_train, y_train)

    log.info("Generating validation predictions...")
    val_pred = pipeline.predict(X_val)
    val_score = pipeline.predict_proba(X_val)[:, 1]

    log.info("Generating test predictions...")
    test_pred = pipeline.predict(X_test)
    test_score = pipeline.predict_proba(X_test)[:, 1]

    val_metrics = compute_basic_metrics(y_val, val_pred)
    test_metrics = compute_basic_metrics(y_test, test_pred)

    log.info("Validation metrics: %s", val_metrics)
    log.info("Test metrics: %s", test_metrics)

    predictions_df = test_df[["scenario", "label_binary"]].copy()
    predictions_df["y_pred"] = test_pred
    predictions_df["y_score"] = test_score

    predictions_path = out_dir / "rf_test_predictions.parquet"
    predictions_df.to_parquet(predictions_path, index=False)

    model_path = out_dir / "rf_model.joblib"
    joblib.dump(pipeline, model_path)

    run_summary = {
        "model": "RandomForestClassifier",
        "target_col": args.target_col,
        "seed": args.seed,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "features": {
            "numeric": NUMERIC_COLS,
            "categorical": CATEGORICAL_COLS,
        },
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "outputs": {
            "model_path": str(model_path),
            "predictions_path": str(predictions_path),
        },
    }

    save_json(run_summary, out_dir / "rf_run_summary.json")

    log.info("Saved model to %s", model_path)
    log.info("Saved test predictions to %s", predictions_path)
    log.info("Saved run summary to %s", out_dir / "rf_run_summary.json")


if __name__ == "__main__":
    main()