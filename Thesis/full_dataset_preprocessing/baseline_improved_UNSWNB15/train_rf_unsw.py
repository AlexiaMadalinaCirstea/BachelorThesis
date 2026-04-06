import os
import json
import argparse
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)


def load_data(train_path: str, test_path: str):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
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


def make_pipeline(categorical_cols, numeric_cols, seed: int):
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

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=seed,
        n_jobs=-1,
        class_weight=None,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    return pipeline


def save_outputs(
    out_dir: str,
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_score: np.ndarray,
):
    os.makedirs(out_dir, exist_ok=True)

    report_dict = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "binary_f1": float(f1_score(y_test, y_pred, average="binary")),
        "precision_attack": float(precision_score(y_test, y_pred, pos_label=1)),
        "recall_attack": float(recall_score(y_test, y_pred, pos_label=1)),
        "precision_benign": float(precision_score(y_test, y_pred, pos_label=0)),
        "recall_benign": float(recall_score(y_test, y_pred, pos_label=0)),
        "confusion_matrix": cm.tolist(),
    }

    preds_df = pd.DataFrame({
        "y_true": y_test.values,
        "y_pred": y_pred,
        "y_score": y_score,
    })
    preds_df.to_csv(os.path.join(out_dir, "rf_test_predictions.csv"), index=False)

    with open(os.path.join(out_dir, "classification_report.json"), "w") as f:
        json.dump(report_dict, f, indent=2)

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(out_dir, "confusion_matrix.txt"), "w") as f:
        f.write(str(cm))

    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write("UNSW-NB15 Random Forest baseline\n\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    print("\nSaved outputs to:", out_dir)
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Train Random Forest baseline on UNSW-NB15")
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--test_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--target_col", default="label")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Loading data...")
    train_df, test_df = load_data(args.train_path, args.test_path)

    print("Preparing features...")
    X_train, X_test, y_train, y_test, categorical_cols, numeric_cols = build_features(
        train_df, test_df, args.target_col
    )

    print("Categorical columns:", categorical_cols)
    print("Numeric columns:", len(numeric_cols))

    pipeline = make_pipeline(categorical_cols, numeric_cols, args.seed)

    print("Training Random Forest...")
    pipeline.fit(X_train, y_train)

    print("Generating predictions...")
    y_pred = pipeline.predict(X_test)
    y_score = pipeline.predict_proba(X_test)[:, 1]

    save_outputs(args.out_dir, y_test, y_pred, y_score)


if __name__ == "__main__":
    main()