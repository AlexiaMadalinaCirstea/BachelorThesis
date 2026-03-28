from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier


RANDOM_STATE = 42


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "label" not in df.columns:
        raise ValueError(f"'label' column not found in {path}")
    return df


def infer_feature_columns(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str = "label",
    drop_cols: list[str] | None = None,
) -> list[str]:
    drop_cols = drop_cols or []
    forbidden = set([label_col] + drop_cols)

    train_cols = set(train_df.columns) - forbidden
    test_cols = set(test_df.columns) - forbidden

    common = sorted(train_cols & test_cols)

    if not common:
        raise ValueError("No common feature columns found between train and test datasets.")

    return common


def build_model(model_name: str):
    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )

    if model_name == "xgb":
        return XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    raise ValueError(f"Unsupported model_name: {model_name}")


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_attack": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_attack": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_attack": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }


def evaluate_direction(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_name: str,
    test_name: str,
    model_name: str,
    out_dir: Path,
    label_col: str = "label",
    drop_cols: list[str] | None = None,
) -> dict:
    features = infer_feature_columns(train_df, test_df, label_col=label_col, drop_cols=drop_cols)

    X_train = train_df[features].copy()
    y_train = train_df[label_col].copy()

    X_test = test_df[features].copy()
    y_test = test_df[label_col].copy()

    model = build_model(model_name)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = compute_metrics(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    direction_name = f"{train_name}_to_{test_name}_{model_name}"
    direction_dir = out_dir / direction_name
    direction_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    joblib.dump(model, direction_dir / "model.joblib")

    # Save predictions
    pred_df = pd.DataFrame(
        {
            "y_true": y_test.values,
            "y_pred": y_pred,
        }
    )
    pred_df.to_csv(direction_dir / "predictions.csv", index=False)

    # Save used features
    pd.DataFrame({"feature": features}).to_csv(direction_dir / "used_features.csv", index=False)

    # Save confusion matrix
    pd.DataFrame(
        cm,
        index=["true_benign", "true_attack"],
        columns=["pred_benign", "pred_attack"],
    ).to_csv(direction_dir / "confusion_matrix.csv")

    # Save metrics
    payload = {
        "train_dataset": train_name,
        "test_dataset": test_name,
        "model": model_name,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_features": int(len(features)),
        "metrics": metrics,
    }
    with open(direction_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return {
        "direction": f"{train_name}->{test_name}",
        "model": model_name,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_features": int(len(features)),
        **metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-dataset evaluation for IoT-23 and UNSW-NB15.")
    parser.add_argument("--iot_csv", required=True, help="Path to processed IoT-23 CSV")
    parser.add_argument("--unsw_csv", required=True, help="Path to processed UNSW-NB15 CSV")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--models", nargs="+", default=["rf", "xgb"], help="Models to run: rf xgb")
    parser.add_argument("--label_col", default="label", help="Label column name")
    parser.add_argument(
        "--drop_cols",
        nargs="*",
        default=[],
        help="Columns to exclude from features, e.g. timestamp scenario_id",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    iot_df = load_dataset(Path(args.iot_csv))
    unsw_df = load_dataset(Path(args.unsw_csv))

    rows = []

    for model_name in args.models:
        rows.append(
            evaluate_direction(
                train_df=iot_df,
                test_df=unsw_df,
                train_name="iot23",
                test_name="unsw_nb15",
                model_name=model_name,
                out_dir=out_dir,
                label_col=args.label_col,
                drop_cols=args.drop_cols,
            )
        )

        rows.append(
            evaluate_direction(
                train_df=unsw_df,
                test_df=iot_df,
                train_name="unsw_nb15",
                test_name="iot23",
                model_name=model_name,
                out_dir=out_dir,
                label_col=args.label_col,
                drop_cols=args.drop_cols,
            )
        )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_dir / "cross_dataset_summary.csv", index=False)

    print("\nCross-dataset evaluation complete.")
    print(summary_df.to_string(index=False))
    print(f"\nSaved outputs to: {out_dir}")


if __name__ == "__main__":
    main()