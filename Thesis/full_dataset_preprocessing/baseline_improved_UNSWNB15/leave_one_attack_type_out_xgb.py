import os
import json
import argparse
import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier


def load_data(train_path: str, test_path: str) -> pd.DataFrame:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    return df


def get_attack_types(df: pd.DataFrame):
    attack_types = sorted(
        [x for x in df["attack_cat"].dropna().astype(str).unique() if x.lower() != "normal"]
    )
    return attack_types


def make_split(df: pd.DataFrame, held_out_attack: str):
    attack_cat = df["attack_cat"].astype(str)

    benign_mask = attack_cat.str.lower() == "normal"
    heldout_mask = attack_cat == held_out_attack
    other_attack_mask = (~benign_mask) & (~heldout_mask)

    train_df = df.loc[benign_mask | other_attack_mask].copy()
    test_df = df.loc[benign_mask | heldout_mask].copy()

    return train_df, test_df


def build_features(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str):
    drop_cols = [target_col, "attack_cat", "id"]
    drop_cols = [c for c in drop_cols if c in train_df.columns]

    X_train = train_df.drop(columns=drop_cols)
    X_test = test_df.drop(columns=drop_cols)

    y_train = train_df[target_col].astype(int)
    y_test = test_df[target_col].astype(int)

    categorical_cols = [c for c in ["proto", "service", "state"] if c in X_train.columns]
    numeric_cols = [c for c in X_train.columns if c not in categorical_cols]

    return X_train, X_test, y_train, y_test, categorical_cols, numeric_cols


def make_pipeline(categorical_cols, numeric_cols, seed: int, n_jobs: int):
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", encoder, categorical_cols),
            ("num", "passthrough", numeric_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=n_jobs,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def extract_feature_importances(pipeline: Pipeline) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    importances = getattr(model, "feature_importances_", None)

    if importances is None:
        raise AttributeError(f"Model does not expose feature_importances_: {type(model)}")
    if len(feature_names) != len(importances):
        raise ValueError(
            f"Feature name / importance length mismatch: {len(feature_names)} vs {len(importances)}"
        )

    return pd.DataFrame({
        "feature": [str(name) for name in feature_names],
        "importance": np.asarray(importances, dtype=float),
    })


def compute_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "binary_f1": float(f1_score(y_true, y_pred, average="binary")),
        "precision_attack": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_attack": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision_benign": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "recall_benign": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "confusion_matrix": cm.tolist(),
    }


def save_fold_outputs(
    out_dir: str,
    held_out_attack: str,
    y_test,
    y_pred,
    y_score,
    metrics,
    pipeline: Pipeline,
    feature_importances_df: pd.DataFrame,
):
    fold_dir = os.path.join(out_dir, held_out_attack.replace("/", "_"))
    os.makedirs(fold_dir, exist_ok=True)

    preds_df = pd.DataFrame({
        "attack_cat_held_out": held_out_attack,
        "y_true": y_test.values,
        "y_pred": y_pred,
        "y_score": y_score,
    })
    preds_df.to_csv(os.path.join(fold_dir, "predictions.csv"), index=False)

    with open(os.path.join(fold_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(fold_dir, "classification_report.json"), "w") as f:
        json.dump(classification_report(y_test, y_pred, output_dict=True, zero_division=0), f, indent=2)

    with open(os.path.join(fold_dir, "summary.txt"), "w") as f:
        f.write(f"Held-out attack type: {held_out_attack}\n\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    joblib.dump(pipeline, os.path.join(fold_dir, "model.joblib"))
    feature_importances_df.to_csv(os.path.join(fold_dir, "fold_feature_importances.csv"), index=False)


def main():
    parser = argparse.ArgumentParser(description="UNSW-NB15 Leave-One-Attack-Type-Out with XGBoost")
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--test_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--target_col", default="label")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_jobs", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading and combining UNSW train/test...")
    df = load_data(args.train_path, args.test_path)

    attack_types = get_attack_types(df)
    print("Held-out attack types:", attack_types)

    all_results = []

    for attack in attack_types:
        print(f"\n=== Held-out attack: {attack} ===")

        train_df, test_df = make_split(df, attack)
        X_train, X_test, y_train, y_test, categorical_cols, numeric_cols = build_features(
            train_df, test_df, args.target_col
        )

        pipeline = make_pipeline(categorical_cols, numeric_cols, args.seed, args.n_jobs)
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_score = pipeline.predict_proba(X_test)[:, 1]
        feature_importances_df = extract_feature_importances(pipeline)

        metrics = compute_metrics(y_test, y_pred)
        metrics["held_out_attack"] = attack
        metrics["train_size"] = int(len(train_df))
        metrics["test_size"] = int(len(test_df))
        metrics["held_out_attack_support"] = int((test_df["attack_cat"] == attack).sum())

        save_fold_outputs(
            args.out_dir,
            attack,
            y_test,
            y_pred,
            y_score,
            metrics,
            pipeline,
            feature_importances_df,
        )
        all_results.append(metrics)

        print(json.dumps(metrics, indent=2))

    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(os.path.join(args.out_dir, "xgb_leave_one_attack_type_summary.csv"), index=False)

    mean_metrics = {
        "accuracy_mean": float(summary_df["accuracy"].mean()),
        "accuracy_std": float(summary_df["accuracy"].std()),
        "macro_f1_mean": float(summary_df["macro_f1"].mean()),
        "macro_f1_std": float(summary_df["macro_f1"].std()),
        "binary_f1_mean": float(summary_df["binary_f1"].mean()),
        "binary_f1_std": float(summary_df["binary_f1"].std()),
    }

    with open(os.path.join(args.out_dir, "xgb_overall_summary.json"), "w") as f:
        json.dump(mean_metrics, f, indent=2)

    print("\n=== Overall summary ===")
    print(json.dumps(mean_metrics, indent=2))


if __name__ == "__main__":
    main()
