from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


try:
    from ..cross_domain_early_detection.cross_domain_early_detection_common import (
        DEFAULT_FRACTIONS,
        IOT23_META_COLS,
        UNSW_META_COLS,
        build_aligned_frame,
        build_feature_mappings,
        downcast_numeric_columns,
        fraction_to_slug,
        infer_column_types,
        load_alignment_table,
        load_iot23_eval_frame,
        load_iot23_source_train,
        load_unsw_frame,
        maybe_sample_rows,
        normalize_binary_labels,
        normalize_categorical_columns,
        parquet_row_count,
        prefix_iot23_by_scenario,
        prefix_unsw_frame,
        prepare_unsw_eval_frame,
        save_json,
        split_unsw_train_val,
        summarize_iot23_scenarios,
        summarize_unsw_attack_categories,
    )
except ImportError:
    THIS_DIR = Path(__file__).resolve().parent
    EARLY_DETECTION_DIR = THIS_DIR.parent
    if str(EARLY_DETECTION_DIR) not in sys.path:
        sys.path.insert(0, str(EARLY_DETECTION_DIR))
    from cross_domain_early_detection.cross_domain_early_detection_common import (
        DEFAULT_FRACTIONS,
        IOT23_META_COLS,
        UNSW_META_COLS,
        build_aligned_frame,
        build_feature_mappings,
        downcast_numeric_columns,
        fraction_to_slug,
        infer_column_types,
        load_alignment_table,
        load_iot23_eval_frame,
        load_iot23_source_train,
        load_unsw_frame,
        maybe_sample_rows,
        normalize_binary_labels,
        normalize_categorical_columns,
        parquet_row_count,
        prefix_iot23_by_scenario,
        prefix_unsw_frame,
        prepare_unsw_eval_frame,
        save_json,
        split_unsw_train_val,
        summarize_iot23_scenarios,
        summarize_unsw_attack_categories,
    )


TRANSFER_CONDITIONS = ["source_only", "target_only", "transfer_adapted"]
DEFAULT_TARGET_TRAIN_BUDGETS = [1000, 5000, 20000, 50000]


def budget_to_slug(budget: int) -> str:
    if budget >= 1_000_000 and budget % 1_000_000 == 0:
        return f"{budget // 1_000_000}m"
    if budget >= 1_000 and budget % 1_000 == 0:
        return f"{budget // 1_000}k"
    return str(budget)


def build_shared_mlp_preprocessor(
    categorical_cols: list[str],
    numeric_cols: list[str],
) -> ColumnTransformer:
    transformers = []

    if numeric_cols:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_cols))

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
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_mlp_classifier(
    hidden_layers: tuple[int, ...],
    alpha: float,
    batch_size: int,
    seed: int,
) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        alpha=alpha,
        batch_size=batch_size,
        learning_rate_init=0.001,
        max_iter=1,
        shuffle=True,
        warm_start=False,
        random_state=seed,
    )


def fit_preprocessor_from_frames(
    frames: list[pd.DataFrame],
    feature_cols: list[str],
    categorical_cols: list[str],
    numeric_cols: list[str],
) -> ColumnTransformer:
    normalized_frames = [
        normalize_categorical_columns(frame[feature_cols].copy(), categorical_cols)
        for frame in frames
        if len(frame) > 0
    ]
    if not normalized_frames:
        raise ValueError("Cannot fit preprocessor without at least one non-empty frame.")
    fit_df = pd.concat(normalized_frames, ignore_index=True)
    preprocessor = build_shared_mlp_preprocessor(categorical_cols, numeric_cols)
    preprocessor.fit(fit_df)
    return preprocessor


def transform_features(
    df: pd.DataFrame,
    preprocessor: ColumnTransformer,
    feature_cols: list[str],
    categorical_cols: list[str],
) -> np.ndarray:
    feature_df = normalize_categorical_columns(df[feature_cols].copy(), categorical_cols)
    transformed = preprocessor.transform(feature_df)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    return np.asarray(transformed, dtype=np.float32)


def train_mlp_incremental(
    X: np.ndarray,
    y: pd.Series | np.ndarray,
    hidden_layers: tuple[int, ...],
    alpha: float,
    batch_size: int,
    seed: int,
    epochs: int,
    initial_model: MLPClassifier | None = None,
) -> MLPClassifier:
    y_array = np.asarray(y, dtype=np.int8)
    if X.shape[0] == 0:
        raise ValueError("Cannot train MLP on an empty dataset.")

    model = copy.deepcopy(initial_model) if initial_model is not None else build_mlp_classifier(
        hidden_layers=hidden_layers,
        alpha=alpha,
        batch_size=batch_size,
        seed=seed,
    )
    classes = np.array([0, 1], dtype=np.int8)

    for epoch in range(max(1, epochs)):
        if epoch == 0 and initial_model is None:
            model.partial_fit(X, y_array, classes=classes)
        else:
            model.partial_fit(X, y_array)
    return model


class PreprocessedMLPPredictor:
    def __init__(
        self,
        preprocessor: ColumnTransformer,
        model: MLPClassifier,
        feature_cols: list[str],
        categorical_cols: list[str],
    ) -> None:
        self.preprocessor = preprocessor
        self.model = model
        self.feature_cols = feature_cols
        self.categorical_cols = categorical_cols

    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        return transform_features(X, self.preprocessor, self.feature_cols, self.categorical_cols)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self._transform(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._transform(X))


def compute_metrics(y_true: pd.Series, y_pred: pd.Series | np.ndarray) -> dict[str, float | int]:
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    precision_attack, recall_attack, f1_attack, support_attack = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[1],
        average=None,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_attack": float(precision_attack[0]),
        "recall_attack": float(recall_attack[0]),
        "f1_attack": float(f1_attack[0]),
        "attack_support": int(support_attack[0]),
        "false_negatives": int(cm[1, 0]),
        "false_positives": int(cm[0, 1]),
        "true_negatives": int(cm[0, 0]),
        "true_positives": int(cm[1, 1]),
    }


def stratified_sample_binary_rows(df: pd.DataFrame, n_rows: int, seed: int) -> pd.DataFrame:
    if n_rows >= len(df):
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    label_counts = df["label"].value_counts().sort_index()
    allocations: dict[int, int] = {}

    for label_value, count in label_counts.items():
        alloc = int(math.floor(n_rows * (count / len(df))))
        allocations[int(label_value)] = min(int(count), alloc)

    for label_value, count in label_counts.items():
        label_int = int(label_value)
        if allocations[label_int] == 0 and count > 0 and n_rows >= label_counts.size:
            allocations[label_int] = 1

    allocated = sum(allocations.values())
    remainders = {
        int(label_value): (n_rows * (count / len(df))) - allocations[int(label_value)]
        for label_value, count in label_counts.items()
    }

    while allocated < n_rows:
        candidates = [
            label_int
            for label_int, count in label_counts.items()
            if allocations[int(label_int)] < int(count)
        ]
        if not candidates:
            break
        next_label = max(candidates, key=lambda label_int: (remainders[label_int], -label_int))
        allocations[next_label] += 1
        allocated += 1

    sampled_parts = []
    for label_value, group in df.groupby("label", sort=False):
        take = allocations.get(int(label_value), 0)
        if take <= 0:
            continue
        sampled_parts.append(group.sample(n=take, random_state=seed + int(label_value)))

    sampled_df = pd.concat(sampled_parts, ignore_index=False)
    if len(sampled_df) < n_rows:
        remaining = df.drop(sampled_df.index)
        extra = remaining.sample(n=n_rows - len(sampled_df), random_state=seed + 999)
        sampled_df = pd.concat([sampled_df, extra], ignore_index=False)

    return sampled_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def prepare_target_train_subset(
    direction: str,
    iot_data_dir: Path,
    unsw_train_csv: Path,
    budget_rows: int,
    seed: int,
    iot_mapping: dict[str, str],
    unsw_mapping: dict[str, str],
) -> pd.DataFrame:
    if direction == "iot23_to_unsw":
        unsw_cols = sorted(set(unsw_mapping.keys()) | {"label"} | set(UNSW_META_COLS))
        unsw_train_raw = load_unsw_frame(unsw_train_csv, unsw_cols)
        target_raw = stratified_sample_binary_rows(unsw_train_raw, budget_rows, seed)
        return build_aligned_frame(target_raw, unsw_mapping, meta_cols=UNSW_META_COLS)

    iot_cols = sorted(set(iot_mapping.keys()) | {"label"} | set(IOT23_META_COLS))
    target_raw = load_iot23_source_train(
        iot_data_dir / "train.parquet",
        iot_cols,
        budget_rows,
        seed,
    )
    return build_aligned_frame(target_raw, iot_mapping, meta_cols=IOT23_META_COLS)


def evaluate_iot23_condition_split(
    split_name: str,
    df: pd.DataFrame,
    predictor: PreprocessedMLPPredictor,
    fractions: list[float],
    out_dir: Path,
    condition: str,
) -> dict[str, pd.DataFrame]:
    split_dir = out_dir / condition / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    detail_rows = []

    for fraction in fractions:
        prefix_df = prefix_iot23_by_scenario(df, fraction).copy()
        prefix_df["y_score"] = predictor.predict_proba(prefix_df[predictor.feature_cols])[:, 1]
        prefix_df["y_pred"] = predictor.predict(prefix_df[predictor.feature_cols])

        metrics = compute_metrics(prefix_df["label"], prefix_df["y_pred"])
        scenario_df = summarize_iot23_scenarios(prefix_df).assign(
            condition=condition,
            split=split_name,
            fraction=fraction,
        )

        fraction_slug = fraction_to_slug(fraction)
        prefix_df.to_parquet(split_dir / f"predictions_frac_{fraction_slug}.parquet", index=False)
        scenario_df.to_csv(split_dir / f"scenario_metrics_frac_{fraction_slug}.csv", index=False)

        summary_rows.append(
            {
                "condition": condition,
                "split": split_name,
                "fraction": fraction,
                "rows_evaluated": int(len(prefix_df)),
                "n_scenarios": int(prefix_df["scenario"].nunique()),
                **metrics,
            }
        )
        detail_rows.append(scenario_df)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(split_dir / "fraction_summary.csv", index=False)

    detail_df = pd.concat(detail_rows, ignore_index=True)
    detail_df.to_csv(split_dir / "scenario_metrics_all_fractions.csv", index=False)
    return {"summary": summary_df, "details": detail_df}


def evaluate_unsw_condition_split(
    split_name: str,
    df: pd.DataFrame,
    predictor: PreprocessedMLPPredictor,
    fractions: list[float],
    out_dir: Path,
    condition: str,
) -> dict[str, pd.DataFrame]:
    split_dir = out_dir / condition / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    detail_rows = []

    for fraction in fractions:
        prefix_df = prefix_unsw_frame(df, fraction).copy()
        prefix_df["y_score"] = predictor.predict_proba(prefix_df[predictor.feature_cols])[:, 1]
        prefix_df["y_pred"] = predictor.predict(prefix_df[predictor.feature_cols])

        metrics = compute_metrics(prefix_df["label"], prefix_df["y_pred"])
        attack_cat_df = summarize_unsw_attack_categories(prefix_df).assign(
            condition=condition,
            split=split_name,
            fraction=fraction,
        )

        fraction_slug = fraction_to_slug(fraction)
        prefix_df.to_parquet(split_dir / f"predictions_frac_{fraction_slug}.parquet", index=False)
        attack_cat_df.to_csv(split_dir / f"attack_cat_metrics_frac_{fraction_slug}.csv", index=False)

        summary_rows.append(
            {
                "condition": condition,
                "split": split_name,
                "fraction": fraction,
                "rows_evaluated": int(len(prefix_df)),
                "n_attack_categories": int(prefix_df["attack_cat"].nunique()),
                **metrics,
            }
        )
        detail_rows.append(attack_cat_df)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(split_dir / "fraction_summary.csv", index=False)

    detail_df = pd.concat(detail_rows, ignore_index=True)
    detail_df.to_csv(split_dir / "attack_cat_metrics_all_fractions.csv", index=False)
    return {"summary": summary_df, "details": detail_df}
