from __future__ import annotations

import argparse
import json
from pathlib import Path


import pandas as pd
import pyarrow.parquet as pq
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-domain shift experiments using the curated aligned IoT-23/UNSW-NB15 feature subset."
    )
    parser.add_argument(
        "--iot_train",
        default="Datasets/IoT23/processed_full/iot23/train.parquet",
        help="Path to IoT-23 train parquet.",
    )
    parser.add_argument(
        "--iot_test",
        default="Datasets/IoT23/processed_full/iot23/test.parquet",
        help="Path to IoT-23 test parquet.",
    )
    parser.add_argument(
        "--unsw_train",
        default="Datasets/UNSW-NB15/UNSW-NB15 dataset/CSV Files/Training and Testing Sets/UNSW_NB15_training-set.csv",
        help="Path to UNSW-NB15 training CSV.",
    )
    parser.add_argument(
        "--unsw_test",
        default="Datasets/UNSW-NB15/UNSW-NB15 dataset/CSV Files/Training and Testing Sets/UNSW_NB15_testing-set.csv",
        help="Path to UNSW-NB15 testing CSV.",
    )
    parser.add_argument(
        "--alignment_csv",
        default="feature_alignment/comparison_outputs/aligned_features_curated.csv",
        help="Path to curated alignment CSV.",
    )
    parser.add_argument(
        "--out_dir",
        default="cross_domain_shift/outputs",
        help="Directory for experiment outputs.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["rf"],
        choices=["rf", "xgb"],
        help="Models to run. Start with rf for safer memory use.",
    )
    parser.add_argument(
        "--include_review_features",
        action="store_true",
        help="Include rows marked review_required in the aligned feature set.",
    )
    parser.add_argument(
        "--iot_train_max_rows",
        type=int,
        default=None,
        help="Optional cap on IoT-23 training rows.",
    )
    parser.add_argument(
        "--iot_test_max_rows",
        type=int,
        default=None,
        help="Optional cap on IoT-23 test rows.",
    )
    parser.add_argument(
        "--unsw_train_max_rows",
        type=int,
        default=None,
        help="Optional cap on UNSW-NB15 training rows.",
    )
    parser.add_argument(
        "--unsw_test_max_rows",
        type=int,
        default=None,
        help="Optional cap on UNSW-NB15 test rows.",
    )
    parser.add_argument(
        "--rf_n_estimators",
        type=int,
        default=150,
        help="Number of trees for Random Forest.",
    )
    parser.add_argument(
        "--rf_max_depth",
        type=int,
        default=12,
        help="Optional max depth for Random Forest. Use smaller values to reduce memory use.",
    )
    parser.add_argument(
        "--xgb_n_estimators",
        type=int,
        default=150,
        help="Number of trees for XGBoost.",
    )
    parser.add_argument(
        "--xgb_max_depth",
        type=int,
        default=6,
        help="Max depth for XGBoost.",
    )
    return parser.parse_args()


def load_alignment_table(path: Path, include_review_features: bool) -> pd.DataFrame:
    df = pd.read_csv(path)

    required_cols = {"aligned_feature", "iot23_feature", "unsw_feature", "status"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Alignment CSV is missing required columns: {missing}")

    allowed_statuses = {"accepted", "accepted_with_normalization"}
    if include_review_features:
        allowed_statuses.add("review_required")

    selected = df[df["status"].isin(allowed_statuses)].copy()
    if selected.empty:
        raise ValueError("No aligned features selected from the curated alignment table.")

    return selected


def maybe_sample_rows(df: pd.DataFrame, max_rows: int | None) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df

    return df.sample(n=max_rows, random_state=RANDOM_STATE).reset_index(drop=True)


def normalize_binary_labels(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="raise").astype("int8")

    normalized = (
        series.astype("string")
        .str.strip()
        .str.lower()
    )

    mapping = {
        "0": 0,
        "1": 1,
        "benign": 0,
        "normal": 0,
        "background": 0,
        "malicious": 1,
        "attack": 1,
        "anomaly": 1,
    }

    mapped = normalized.map(mapping)
    unknown = sorted(normalized[mapped.isna()].dropna().unique().tolist())
    if unknown:
        raise ValueError(f"Unsupported label values encountered: {unknown}")

    return mapped.astype("int8")


def downcast_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    int_cols = df.select_dtypes(include=["int", "int64", "int32"]).columns
    float_cols = df.select_dtypes(include=["float", "float64", "float32"]).columns

    for col in int_cols:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    for col in float_cols:
        df[col] = pd.to_numeric(df[col], downcast="float")

    return df


def normalize_categorical_columns(df: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    df = df.copy()

    for col in categorical_cols:
        df[col] = df[col].astype("string").fillna("missing")

    return df


def load_iot23(path: Path, columns: list[str], max_rows: int | None) -> pd.DataFrame:
    table = pq.read_table(path, columns=columns)
    df = table.to_pandas()
    df = maybe_sample_rows(df, max_rows=max_rows)
    df["label"] = normalize_binary_labels(df["label"])
    df = downcast_numeric_columns(df)
    return df


def load_unsw(path: Path, columns: list[str], max_rows: int | None) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=columns)
    df = maybe_sample_rows(df, max_rows=max_rows)
    df["label"] = normalize_binary_labels(df["label"])
    df = downcast_numeric_columns(df)
    return df


def build_aligned_views(
    iot_df: pd.DataFrame,
    unsw_df: pd.DataFrame,
    alignment_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, str], dict[str, str]]:
    iot_mapping = dict(zip(alignment_df["iot23_feature"], alignment_df["aligned_feature"]))
    unsw_mapping = dict(zip(alignment_df["unsw_feature"], alignment_df["aligned_feature"]))

    missing_iot = [col for col in iot_mapping if col not in iot_df.columns]
    missing_unsw = [col for col in unsw_mapping if col not in unsw_df.columns]

    if missing_iot:
        raise ValueError(f"IoT-23 is missing aligned columns: {missing_iot}")
    if missing_unsw:
        raise ValueError(f"UNSW-NB15 is missing aligned columns: {missing_unsw}")

    iot_aligned = iot_df[list(iot_mapping.keys()) + ["label"]].rename(columns=iot_mapping).copy()
    unsw_aligned = unsw_df[list(unsw_mapping.keys()) + ["label"]].rename(columns=unsw_mapping).copy()

    feature_cols = [col for col in alignment_df["aligned_feature"].tolist() if col in iot_aligned.columns]
    return iot_aligned, unsw_aligned, feature_cols, iot_mapping, unsw_mapping


def infer_column_types(df: pd.DataFrame, feature_cols: list[str]) -> tuple[list[str], list[str]]:
    categorical_cols = []
    numeric_cols = []

    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return categorical_cols, numeric_cols


def build_model(model_name: str, args: argparse.Namespace):
    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=args.rf_n_estimators,
            max_depth=args.rf_max_depth,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )

    if model_name == "xgb":
        if XGBClassifier is None:
            raise ImportError("xgboost is not installed, but model 'xgb' was requested.")

        return XGBClassifier(
            n_estimators=args.xgb_n_estimators,
            max_depth=args.xgb_max_depth,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    raise ValueError(f"Unsupported model: {model_name}")


def build_pipeline(model_name: str, categorical_cols: list[str], numeric_cols: list[str], args: argparse.Namespace) -> Pipeline:
    transformers = []

    if categorical_cols:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                ),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_cols))

    if numeric_cols:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    model = build_model(model_name, args=args)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def compute_metrics(y_true: pd.Series, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_attack": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_attack": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_attack": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }


def save_feature_importance(pipeline: Pipeline, feature_cols: list[str], out_path: Path) -> None:
    model = pipeline.named_steps["model"]

    if not hasattr(model, "feature_importances_"):
        return

    importances = model.feature_importances_
    if len(importances) != len(feature_cols):
        return

    pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False).to_csv(out_path, index=False)


def evaluate_direction(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    train_name: str,
    test_name: str,
    model_name: str,
    out_dir: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    X_train = train_df[feature_cols].copy()
    y_train = train_df["label"].copy()
    X_test = test_df[feature_cols].copy()
    y_test = test_df["label"].copy()

    categorical_cols, numeric_cols = infer_column_types(train_df, feature_cols)
    X_train = normalize_categorical_columns(X_train, categorical_cols)
    X_test = normalize_categorical_columns(X_test, categorical_cols)

    pipeline = build_pipeline(
        model_name=model_name,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
        args=args,
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    direction_dir = out_dir / f"{train_name}_to_{test_name}" / model_name
    direction_dir.mkdir(parents=True, exist_ok=True)

    #joblib.dump(pipeline, direction_dir / "model.joblib")
    pd.DataFrame({"feature": feature_cols}).to_csv(direction_dir / "used_features.csv", index=False)
    pd.DataFrame({"y_true": y_test.values, "y_pred": y_pred}).to_csv(
        direction_dir / "predictions.csv",
        index=False,
    )
    pd.DataFrame(
        cm,
        index=["true_benign", "true_attack"],
        columns=["pred_benign", "pred_attack"],
    ).to_csv(direction_dir / "confusion_matrix.csv")

    save_feature_importance(
        pipeline=pipeline,
        feature_cols=feature_cols,
        out_path=direction_dir / "feature_importance.csv",
    )

    payload = {
        "train_dataset": train_name,
        "test_dataset": test_name,
        "model": model_name,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_features": int(len(feature_cols)),
        "categorical_features": categorical_cols,
        "numeric_features": numeric_cols,
        "metrics": metrics,
    }

    with open(direction_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return {
        "direction": f"{train_name}->{test_name}",
        "model": model_name,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_features": int(len(feature_cols)),
        **metrics,
    }


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    alignment_df = load_alignment_table(
        Path(args.alignment_csv),
        include_review_features=args.include_review_features,
    )

    iot_required_cols = sorted(set(alignment_df["iot23_feature"].tolist() + ["label"]))
    unsw_required_cols = sorted(set(alignment_df["unsw_feature"].tolist() + ["label"]))

    iot_train = load_iot23(Path(args.iot_train), columns=iot_required_cols, max_rows=args.iot_train_max_rows)
    iot_test = load_iot23(Path(args.iot_test), columns=iot_required_cols, max_rows=args.iot_test_max_rows)
    unsw_train = load_unsw(Path(args.unsw_train), columns=unsw_required_cols, max_rows=args.unsw_train_max_rows)
    unsw_test = load_unsw(Path(args.unsw_test), columns=unsw_required_cols, max_rows=args.unsw_test_max_rows)

    iot_train_aligned, unsw_train_aligned, feature_cols, iot_mapping, unsw_mapping = build_aligned_views(
        iot_df=iot_train,
        unsw_df=unsw_train,
        alignment_df=alignment_df,
    )
    iot_test_aligned, unsw_test_aligned, _, _, _ = build_aligned_views(
        iot_df=iot_test,
        unsw_df=unsw_test,
        alignment_df=alignment_df,
    )

    config = {
        "iot_train": args.iot_train,
        "iot_test": args.iot_test,
        "unsw_train": args.unsw_train,
        "unsw_test": args.unsw_test,
        "alignment_csv": args.alignment_csv,
        "include_review_features": args.include_review_features,
        "models": args.models,
        "iot_train_max_rows": args.iot_train_max_rows,
        "iot_test_max_rows": args.iot_test_max_rows,
        "unsw_train_max_rows": args.unsw_train_max_rows,
        "unsw_test_max_rows": args.unsw_test_max_rows,
        "rf_n_estimators": args.rf_n_estimators,
        "rf_max_depth": args.rf_max_depth,
        "xgb_n_estimators": args.xgb_n_estimators,
        "xgb_max_depth": args.xgb_max_depth,
        "n_aligned_features": len(feature_cols),
        "aligned_features": feature_cols,
        "iot23_to_aligned_mapping": iot_mapping,
        "unsw_to_aligned_mapping": unsw_mapping,
    }

    with open(out_dir / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    summary_rows = []

    for model_name in args.models:
        summary_rows.append(
            evaluate_direction(
                train_df=iot_train_aligned,
                test_df=unsw_test_aligned,
                feature_cols=feature_cols,
                train_name="iot23_train",
                test_name="unsw_test",
                model_name=model_name,
                out_dir=out_dir,
                args=args,
            )
        )
        summary_rows.append(
            evaluate_direction(
                train_df=unsw_train_aligned,
                test_df=iot_test_aligned,
                feature_cols=feature_cols,
                train_name="unsw_train",
                test_name="iot23_test",
                model_name=model_name,
                out_dir=out_dir,
                args=args,
            )
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "cross_domain_shift_summary.csv", index=False)

    print("Cross-domain shift setup complete.")
    print(summary_df.to_string(index=False))
    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
