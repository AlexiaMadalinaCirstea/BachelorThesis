from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier


RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Transfer-learning experiments for IoT-23 and UNSW-NB15 using the curated aligned feature subset. "
            "The script compares source-only transfer, target-only training, and staged source-to-target adaptation."
        )
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
        default="transfer_learning/outputs",
        help="Directory for experiment outputs.",
    )
    parser.add_argument(
        "--include_review_features",
        action="store_true",
        help="Include rows marked review_required in the aligned feature set.",
    )
    parser.add_argument(
        "--target_fractions",
        nargs="+",
        type=float,
        default=[0.05, 0.10, 0.25, 0.50, 1.0],
        help="Fractions of target-train data used for target-only and transfer-learning adaptation.",
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
        "--pretrain_estimators",
        type=int,
        default=150,
        help="Number of trees for the source pretraining stage.",
    )
    parser.add_argument(
        "--adapt_estimators",
        type=int,
        default=75,
        help="Additional trees added during target adaptation.",
    )
    parser.add_argument(
        "--target_only_estimators",
        type=int,
        default=150,
        help="Number of trees for target-only training from scratch.",
    )
    parser.add_argument(
        "--xgb_max_depth",
        type=int,
        default=6,
        help="Max depth for XGBoost.",
    )
    parser.add_argument(
        "--xgb_learning_rate",
        type=float,
        default=0.05,
        help="Learning rate for XGBoost.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=RANDOM_STATE,
        help="Random seed for row sampling and model training.",
    )
    parser.add_argument(
        "--balance_target_train",
        action="store_true",
        help="Apply balanced sampling to the target-train subset before target-only and transfer-learning runs.",
    )
    parser.add_argument(
        "--target_balance_ratio",
        type=float,
        default=1.0,
        help=(
            "Desired majority-to-minority ratio after target balancing. "
            "Use 1.0 for 1:1 balancing, 2.0 for 2:1, etc."
        ),
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


def maybe_sample_rows(df: pd.DataFrame, max_rows: int | None, random_state: int) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df.reset_index(drop=True)

    return df.sample(n=max_rows, random_state=random_state).reset_index(drop=True)


def sample_target_fraction(df: pd.DataFrame, fraction: float, random_state: int) -> pd.DataFrame:
    if fraction <= 0 or fraction > 1:
        raise ValueError(f"Target fraction must be in (0, 1], received {fraction}")

    if fraction == 1.0:
        return df.reset_index(drop=True)

    sampled_parts = []
    for _, group in df.groupby("label", sort=True):
        n = max(1, int(round(len(group) * fraction)))
        n = min(n, len(group))
        sampled_parts.append(group.sample(n=n, random_state=random_state))

    sampled = pd.concat(sampled_parts, axis=0)
    sampled = sampled.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return sampled


def balance_binary_target_train(
    df: pd.DataFrame,
    random_state: int,
    majority_to_minority_ratio: float,
) -> pd.DataFrame:
    if "label" not in df.columns:
        raise ValueError("Target balancing requires a 'label' column.")

    if majority_to_minority_ratio < 1.0:
        raise ValueError(
            f"target_balance_ratio must be >= 1.0, received {majority_to_minority_ratio}"
        )

    counts = df["label"].value_counts()
    if len(counts) < 2:
        return df.reset_index(drop=True)

    minority_label = counts.idxmin()
    majority_label = counts.idxmax()
    minority_df = df[df["label"] == minority_label]
    majority_df = df[df["label"] == majority_label]

    minority_n = len(minority_df)
    majority_target_n = min(len(majority_df), int(round(minority_n * majority_to_minority_ratio)))
    majority_target_n = max(1, majority_target_n)

    majority_sampled = majority_df.sample(n=majority_target_n, random_state=random_state)
    balanced = pd.concat([minority_df, majority_sampled], axis=0)
    balanced = balanced.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return balanced


def normalize_binary_labels(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="raise").astype("int8")

    normalized = series.astype("string").str.strip().str.lower()
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


def load_iot23(path: Path, columns: list[str], max_rows: int | None, random_state: int) -> pd.DataFrame:
    table = pq.read_table(path, columns=columns)
    df = table.to_pandas()
    df = maybe_sample_rows(df, max_rows=max_rows, random_state=random_state)
    df["label"] = normalize_binary_labels(df["label"])
    return downcast_numeric_columns(df)


def load_unsw(path: Path, columns: list[str], max_rows: int | None, random_state: int) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=columns)
    df = maybe_sample_rows(df, max_rows=max_rows, random_state=random_state)
    df["label"] = normalize_binary_labels(df["label"])
    return downcast_numeric_columns(df)


def build_aligned_views(
    iot_df: pd.DataFrame,
    unsw_df: pd.DataFrame,
    alignment_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
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
    return iot_aligned, unsw_aligned, feature_cols


def infer_column_types(df: pd.DataFrame, feature_cols: list[str]) -> tuple[list[str], list[str]]:
    categorical_cols = []
    numeric_cols = []
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    return categorical_cols, numeric_cols


def normalize_categorical_columns(df: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in categorical_cols:
        df[col] = df[col].astype("string").fillna("missing")
    return df


def build_preprocessor(categorical_cols: list[str], numeric_cols: list[str]) -> ColumnTransformer:
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

    return ColumnTransformer(transformers=transformers, remainder="drop")


def transformed_feature_names(categorical_cols: list[str], numeric_cols: list[str]) -> list[str]:
    return list(categorical_cols) + list(numeric_cols)


def build_xgb_model(n_estimators: int, args: argparse.Namespace) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=args.xgb_max_depth,
        learning_rate=args.xgb_learning_rate,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=args.random_state,
        n_jobs=-1,
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


def save_feature_importance(model: XGBClassifier, feature_names: list[str], out_path: Path) -> None:
    if not hasattr(model, "feature_importances_"):
        return

    importances = model.feature_importances_
    if len(importances) != len(feature_names):
        return

    pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(
        "importance",
        ascending=False,
    ).to_csv(out_path, index=False)


def save_run_artifacts(
    out_dir: Path,
    y_true: pd.Series,
    y_pred,
    y_score,
    metrics: dict[str, float],
    feature_names: list[str],
    model: XGBClassifier,
    metadata: dict[str, object],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "y_true": y_true.values,
            "y_pred": y_pred,
            "y_score_attack": y_score,
        }
    ).to_csv(out_dir / "predictions.csv", index=False)
    cm = confusion_matrix(y_true, y_pred)
    pd.DataFrame(
        cm,
        index=["true_benign", "true_attack"],
        columns=["pred_benign", "pred_attack"],
    ).to_csv(out_dir / "confusion_matrix.csv")

    save_feature_importance(model=model, feature_names=feature_names, out_path=out_dir / "feature_importance.csv")

    payload = dict(metadata)
    payload["metrics"] = metrics
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def prepare_matrix_views(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
    numeric_cols: list[str],
) -> tuple[ColumnTransformer, object, object, list[str]]:
    X_train = normalize_categorical_columns(train_df[feature_cols].copy(), categorical_cols)
    X_test = normalize_categorical_columns(test_df[feature_cols].copy(), categorical_cols)

    preprocessor = build_preprocessor(categorical_cols=categorical_cols, numeric_cols=numeric_cols)
    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)
    feature_names = transformed_feature_names(categorical_cols=categorical_cols, numeric_cols=numeric_cols)
    return preprocessor, X_train_encoded, X_test_encoded, feature_names


def encode_with_existing_preprocessor(
    preprocessor: ColumnTransformer,
    df: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
) -> object:
    X = normalize_categorical_columns(df[feature_cols].copy(), categorical_cols)
    return preprocessor.transform(X)


def evaluate_source_only(
    source_train_df: pd.DataFrame,
    target_test_df: pd.DataFrame,
    feature_cols: list[str],
    source_name: str,
    target_name: str,
    direction_dir: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    categorical_cols, numeric_cols = infer_column_types(source_train_df, feature_cols)
    _, X_source, X_target_test, feature_names = prepare_matrix_views(
        train_df=source_train_df,
        test_df=target_test_df,
        feature_cols=feature_cols,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
    )

    model = build_xgb_model(n_estimators=args.pretrain_estimators, args=args)
    y_source = source_train_df["label"].copy()
    y_target_test = target_test_df["label"].copy()
    model.fit(X_source, y_source)

    y_pred = model.predict(X_target_test)
    y_score = model.predict_proba(X_target_test)[:, 1]
    metrics = compute_metrics(y_target_test, y_pred)

    run_dir = direction_dir / "source_only"
    save_run_artifacts(
        out_dir=run_dir,
        y_true=y_target_test,
        y_pred=y_pred,
        y_score=y_score,
        metrics=metrics,
        feature_names=feature_names,
        model=model,
        metadata={
            "condition": "source_only",
            "source_dataset": source_name,
            "target_dataset": target_name,
            "n_source_train": int(len(source_train_df)),
            "n_target_test": int(len(target_test_df)),
            "n_features": int(len(feature_cols)),
            "categorical_features": categorical_cols,
            "numeric_features": numeric_cols,
            "pretrain_estimators": args.pretrain_estimators,
        },
    )

    return {
        "direction": f"{source_name}->{target_name}",
        "condition": "source_only",
        "target_fraction": 0.0,
        "n_source_train": int(len(source_train_df)),
        "n_target_train": 0,
        "n_target_test": int(len(target_test_df)),
        "n_features": int(len(feature_cols)),
        **metrics,
    }


def evaluate_target_only(
    target_train_df: pd.DataFrame,
    target_test_df: pd.DataFrame,
    feature_cols: list[str],
    direction_label: str,
    fraction: float,
    direction_dir: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    categorical_cols, numeric_cols = infer_column_types(target_train_df, feature_cols)
    _, X_target_train, X_target_test, feature_names = prepare_matrix_views(
        train_df=target_train_df,
        test_df=target_test_df,
        feature_cols=feature_cols,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
    )

    model = build_xgb_model(n_estimators=args.target_only_estimators, args=args)
    y_target_train = target_train_df["label"].copy()
    y_target_test = target_test_df["label"].copy()
    model.fit(X_target_train, y_target_train)

    y_pred = model.predict(X_target_test)
    y_score = model.predict_proba(X_target_test)[:, 1]
    metrics = compute_metrics(y_target_test, y_pred)

    fraction_slug = fraction_to_slug(fraction)
    run_dir = direction_dir / f"target_only_{fraction_slug}"
    save_run_artifacts(
        out_dir=run_dir,
        y_true=y_target_test,
        y_pred=y_pred,
        y_score=y_score,
        metrics=metrics,
        feature_names=feature_names,
        model=model,
        metadata={
            "condition": "target_only",
            "direction": direction_label,
            "target_fraction": fraction,
            "n_target_train": int(len(target_train_df)),
            "n_target_test": int(len(target_test_df)),
            "n_features": int(len(feature_cols)),
            "categorical_features": categorical_cols,
            "numeric_features": numeric_cols,
            "target_only_estimators": args.target_only_estimators,
        },
    )

    return {
        "direction": direction_label,
        "condition": "target_only",
        "target_fraction": fraction,
        "n_source_train": 0,
        "n_target_train": int(len(target_train_df)),
        "n_target_test": int(len(target_test_df)),
        "n_features": int(len(feature_cols)),
        **metrics,
    }


def evaluate_transfer_learning(
    source_train_df: pd.DataFrame,
    target_train_df: pd.DataFrame,
    target_test_df: pd.DataFrame,
    feature_cols: list[str],
    direction_label: str,
    source_name: str,
    target_name: str,
    fraction: float,
    direction_dir: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    categorical_cols, numeric_cols = infer_column_types(source_train_df, feature_cols)

    X_source_df = normalize_categorical_columns(source_train_df[feature_cols].copy(), categorical_cols)
    X_target_train_df = normalize_categorical_columns(target_train_df[feature_cols].copy(), categorical_cols)
    X_target_test_df = normalize_categorical_columns(target_test_df[feature_cols].copy(), categorical_cols)

    preprocessor = build_preprocessor(categorical_cols=categorical_cols, numeric_cols=numeric_cols)
    X_source = preprocessor.fit_transform(X_source_df)
    X_target_train = preprocessor.transform(X_target_train_df)
    X_target_test = preprocessor.transform(X_target_test_df)
    feature_names = transformed_feature_names(categorical_cols=categorical_cols, numeric_cols=numeric_cols)

    y_source = source_train_df["label"].copy()
    y_target_train = target_train_df["label"].copy()
    y_target_test = target_test_df["label"].copy()

    source_model = build_xgb_model(n_estimators=args.pretrain_estimators, args=args)
    source_model.fit(X_source, y_source)

    adapted_model = build_xgb_model(n_estimators=args.adapt_estimators, args=args)
    adapted_model.fit(X_target_train, y_target_train, xgb_model=source_model.get_booster())

    y_pred = adapted_model.predict(X_target_test)
    y_score = adapted_model.predict_proba(X_target_test)[:, 1]
    metrics = compute_metrics(y_target_test, y_pred)

    fraction_slug = fraction_to_slug(fraction)
    run_dir = direction_dir / f"transfer_learning_{fraction_slug}"
    save_run_artifacts(
        out_dir=run_dir,
        y_true=y_target_test,
        y_pred=y_pred,
        y_score=y_score,
        metrics=metrics,
        feature_names=feature_names,
        model=adapted_model,
        metadata={
            "condition": "transfer_learning",
            "direction": direction_label,
            "source_dataset": source_name,
            "target_dataset": target_name,
            "target_fraction": fraction,
            "n_source_train": int(len(source_train_df)),
            "n_target_train": int(len(target_train_df)),
            "n_target_test": int(len(target_test_df)),
            "n_features": int(len(feature_cols)),
            "categorical_features": categorical_cols,
            "numeric_features": numeric_cols,
            "pretrain_estimators": args.pretrain_estimators,
            "adapt_estimators": args.adapt_estimators,
        },
    )
    save_feature_importance(source_model, feature_names, run_dir / "feature_importance_pretrain.csv")

    return {
        "direction": direction_label,
        "condition": "transfer_learning",
        "target_fraction": fraction,
        "n_source_train": int(len(source_train_df)),
        "n_target_train": int(len(target_train_df)),
        "n_target_test": int(len(target_test_df)),
        "n_features": int(len(feature_cols)),
        **metrics,
    }


def fraction_to_slug(fraction: float) -> str:
    return f"frac_{str(fraction).replace('.', 'p')}"


def direction_to_slug(source_name: str, target_name: str) -> str:
    return f"{source_name}_to_{target_name}_test"


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

    iot_train = load_iot23(Path(args.iot_train), columns=iot_required_cols, max_rows=args.iot_train_max_rows, random_state=args.random_state)
    iot_test = load_iot23(Path(args.iot_test), columns=iot_required_cols, max_rows=args.iot_test_max_rows, random_state=args.random_state)
    unsw_train = load_unsw(Path(args.unsw_train), columns=unsw_required_cols, max_rows=args.unsw_train_max_rows, random_state=args.random_state)
    unsw_test = load_unsw(Path(args.unsw_test), columns=unsw_required_cols, max_rows=args.unsw_test_max_rows, random_state=args.random_state)

    iot_train_aligned, unsw_train_aligned, feature_cols = build_aligned_views(
        iot_df=iot_train,
        unsw_df=unsw_train,
        alignment_df=alignment_df,
    )
    iot_test_aligned, unsw_test_aligned, _ = build_aligned_views(
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
        "target_fractions": args.target_fractions,
        "iot_train_max_rows": args.iot_train_max_rows,
        "iot_test_max_rows": args.iot_test_max_rows,
        "unsw_train_max_rows": args.unsw_train_max_rows,
        "unsw_test_max_rows": args.unsw_test_max_rows,
        "pretrain_estimators": args.pretrain_estimators,
        "adapt_estimators": args.adapt_estimators,
        "target_only_estimators": args.target_only_estimators,
        "xgb_max_depth": args.xgb_max_depth,
        "xgb_learning_rate": args.xgb_learning_rate,
        "random_state": args.random_state,
        "balance_target_train": args.balance_target_train,
        "target_balance_ratio": args.target_balance_ratio,
        "n_aligned_features": len(feature_cols),
        "aligned_features": feature_cols,
    }
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    summary_rows = []
    directions = [
        ("iot23_train", "unsw", iot_train_aligned, unsw_train_aligned, unsw_test_aligned),
        ("unsw_train", "iot23", unsw_train_aligned, iot_train_aligned, iot_test_aligned),
    ]

    for source_name, target_name, source_train_df, target_train_df, target_test_df in directions:
        direction_label = f"{source_name}->{target_name}_test"
        direction_slug = direction_to_slug(source_name=source_name, target_name=target_name)
        direction_dir = out_dir / direction_slug
        direction_dir.mkdir(parents=True, exist_ok=True)

        summary_rows.append(
            evaluate_source_only(
                source_train_df=source_train_df,
                target_test_df=target_test_df,
                feature_cols=feature_cols,
                source_name=source_name,
                target_name=f"{target_name}_test",
                direction_dir=direction_dir,
                args=args,
            )
        )

        for fraction in args.target_fractions:
            sampled_target_train = sample_target_fraction(
                df=target_train_df,
                fraction=fraction,
                random_state=args.random_state,
            )
            if args.balance_target_train:
                sampled_target_train = balance_binary_target_train(
                    df=sampled_target_train,
                    random_state=args.random_state,
                    majority_to_minority_ratio=args.target_balance_ratio,
                )

            summary_rows.append(
                evaluate_target_only(
                    target_train_df=sampled_target_train,
                    target_test_df=target_test_df,
                    feature_cols=feature_cols,
                    direction_label=direction_label,
                    fraction=fraction,
                    direction_dir=direction_dir,
                    args=args,
                )
            )

            summary_rows.append(
                evaluate_transfer_learning(
                    source_train_df=source_train_df,
                    target_train_df=sampled_target_train,
                    target_test_df=target_test_df,
                    feature_cols=feature_cols,
                    direction_label=direction_label,
                    source_name=source_name,
                    target_name=f"{target_name}_test",
                    fraction=fraction,
                    direction_dir=direction_dir,
                    args=args,
                )
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "transfer_learning_summary.csv", index=False)

    print("Transfer-learning setup complete.")
    print(summary_df.to_string(index=False))
    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
