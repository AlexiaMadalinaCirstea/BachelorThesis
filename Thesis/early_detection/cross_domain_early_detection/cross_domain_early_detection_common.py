from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


DEFAULT_FRACTIONS = [0.02, 0.05, 0.10, 0.20, 0.50, 1.00]
IOT23_FULL_LOAD_ROW_GUARD = 5_000_000

IOT23_META_COLS = [
    "scenario",
    "split",
    "ts",
    "label_phase",
    "label_binary",
]

UNSW_META_COLS = [
    "id",
    "attack_cat",
]


def save_json(obj: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True)


def fraction_to_slug(fraction: float) -> str:
    return str(fraction).replace(".", "p")


def load_alignment_table(path: Path, include_review_features: bool) -> pd.DataFrame:
    df = pd.read_csv(path)

    required_cols = {"aligned_feature", "iot23_feature", "unsw_feature", "status"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Alignment CSV is missing required columns: {sorted(missing)}")

    allowed_statuses = {"accepted", "accepted_with_normalization"}
    if include_review_features:
        allowed_statuses.add("review_required")

    selected = df[df["status"].isin(allowed_statuses)].copy()
    if selected.empty:
        raise ValueError("No aligned features selected from the curated alignment table.")

    return selected


def build_feature_mappings(alignment_df: pd.DataFrame) -> tuple[dict[str, str], dict[str, str], list[str]]:
    iot_mapping = dict(zip(alignment_df["iot23_feature"], alignment_df["aligned_feature"]))
    unsw_mapping = dict(zip(alignment_df["unsw_feature"], alignment_df["aligned_feature"]))
    feature_cols = alignment_df["aligned_feature"].tolist()
    return iot_mapping, unsw_mapping, feature_cols


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


def maybe_sample_rows(df: pd.DataFrame, max_rows: int | None, seed: int) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df.reset_index(drop=True)
    return df.sample(n=max_rows, random_state=seed).reset_index(drop=True)


def load_iot23_frame(path: Path, columns: list[str]) -> pd.DataFrame:
    table = pq.read_table(path, columns=columns)
    df = table.to_pandas()
    if "label" in df.columns:
        df["label"] = normalize_binary_labels(df["label"])
    return downcast_numeric_columns(df)


def parquet_row_count(path: Path) -> int:
    return int(pq.ParquetFile(path).metadata.num_rows)


def stream_sample_iot23_train(path: Path, columns: list[str], max_rows: int, seed: int) -> pd.DataFrame:
    parquet = pq.ParquetFile(path)
    total_rows = parquet.metadata.num_rows
    if total_rows <= max_rows:
        return load_iot23_frame(path, columns=columns)

    sample_prob = min(1.0, max_rows / total_rows)
    sampled_parts: list[pd.DataFrame] = []

    for batch_index, batch in enumerate(parquet.iter_batches(columns=columns, batch_size=100_000)):
        batch_df = batch.to_pandas()
        sampled_batch = batch_df.sample(frac=sample_prob, random_state=seed + batch_index)
        if not sampled_batch.empty:
            sampled_parts.append(sampled_batch)

    if not sampled_parts:
        fallback = load_iot23_frame(path, columns=columns).head(max_rows)
        return fallback.reset_index(drop=True)

    sampled_df = pd.concat(sampled_parts, ignore_index=True)
    if len(sampled_df) > max_rows:
        sampled_df = sampled_df.sample(n=max_rows, random_state=seed).reset_index(drop=True)

    sampled_df["label"] = normalize_binary_labels(sampled_df["label"])
    return downcast_numeric_columns(sampled_df.reset_index(drop=True))


def load_iot23_source_train(path: Path, columns: list[str], max_rows: int | None, seed: int) -> pd.DataFrame:
    if max_rows is not None:
        return stream_sample_iot23_train(path, columns, max_rows, seed)

    total_rows = parquet_row_count(path)
    if total_rows > IOT23_FULL_LOAD_ROW_GUARD:
        raise ValueError(
            f"IoT-23 source train has {total_rows} rows, which is too large for an uncapped full load. "
            "Set --iot_train_max_rows or use the processed_test_sample dataset first."
        )

    return load_iot23_frame(path, columns=columns)


def load_unsw_frame(path: Path, columns: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=columns)
    if "label" in df.columns:
        df["label"] = normalize_binary_labels(df["label"])
    return downcast_numeric_columns(df)


def build_aligned_frame(
    df: pd.DataFrame,
    feature_mapping: dict[str, str],
    meta_cols: list[str] | None = None,
) -> pd.DataFrame:
    meta_cols = meta_cols or []
    missing = [col for col in feature_mapping if col not in df.columns]
    if missing:
        raise ValueError(f"Missing aligned source columns: {missing}")

    keep_cols = list(feature_mapping.keys()) + ["label"] + [col for col in meta_cols if col in df.columns]
    aligned_df = df[keep_cols].rename(columns=feature_mapping).copy()
    return aligned_df


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


def build_rf_pipeline(
    categorical_cols: list[str],
    numeric_cols: list[str],
    seed: int,
    n_estimators: int,
    max_depth: int | None,
    n_jobs: int,
) -> Pipeline:
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
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
        n_jobs=n_jobs,
        class_weight="balanced_subsample",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def build_mlp_pipeline(
    categorical_cols: list[str],
    numeric_cols: list[str],
    hidden_layers: tuple[int, ...],
    alpha: float,
    max_iter: int,
    batch_size: int,
    seed: int,
) -> Pipeline:
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

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        alpha=alpha,
        batch_size=batch_size,
        learning_rate_init=0.001,
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        random_state=seed,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float | int]:
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


def prepare_iot23_eval_frame(df: pd.DataFrame, max_rows_per_scenario: int | None) -> pd.DataFrame:
    ordered = df.sort_values(["scenario", "ts"], kind="mergesort").reset_index(drop=True)
    if max_rows_per_scenario is None:
        return ordered

    return (
        ordered.groupby("scenario", sort=False, group_keys=False)
        .head(max_rows_per_scenario)
        .reset_index(drop=True)
    )


def load_iot23_eval_frame(path: Path, columns: list[str], max_rows_per_scenario: int | None) -> pd.DataFrame:
    if max_rows_per_scenario is None:
        total_rows = parquet_row_count(path)
        if total_rows > IOT23_FULL_LOAD_ROW_GUARD:
            raise ValueError(
                f"IoT-23 evaluation split {path.name} has {total_rows} rows, which is too large for an uncapped full load. "
                "Set --iot_eval_max_rows_per_scenario or use the processed_test_sample dataset first."
            )
        return prepare_iot23_eval_frame(load_iot23_frame(path, columns=columns), None)

    parquet = pq.ParquetFile(path)
    kept_by_scenario: dict[str, pd.DataFrame] = {}

    for batch in parquet.iter_batches(columns=columns, batch_size=100_000):
        batch_df = batch.to_pandas()
        if "label" in batch_df.columns:
            batch_df["label"] = normalize_binary_labels(batch_df["label"])

        for scenario, scenario_df in batch_df.groupby("scenario", sort=False):
            current = kept_by_scenario.get(scenario)
            if current is None:
                merged = scenario_df
            else:
                merged = pd.concat([current, scenario_df], ignore_index=True)

            kept_by_scenario[scenario] = (
                merged.sort_values("ts", kind="mergesort")
                .head(max_rows_per_scenario)
                .reset_index(drop=True)
            )

    if not kept_by_scenario:
        return pd.DataFrame(columns=columns)

    combined = pd.concat(kept_by_scenario.values(), ignore_index=True)
    combined = combined.sort_values(["scenario", "ts"], kind="mergesort").reset_index(drop=True)
    return downcast_numeric_columns(combined)


def prefix_iot23_by_scenario(df: pd.DataFrame, fraction: float) -> pd.DataFrame:
    if not 0 < fraction <= 1:
        raise ValueError(f"Invalid fraction: {fraction}")

    parts = []
    for _, group in df.groupby("scenario", sort=False):
        keep = max(1, int(len(group) * fraction))
        parts.append(group.iloc[:keep])

    return pd.concat(parts, ignore_index=True)


def summarize_iot23_scenarios(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario, group in pred_df.groupby("scenario", sort=False):
        metrics = compute_metrics(group["label"], group["y_pred"])
        rows.append(
            {
                "scenario": scenario,
                "rows": int(len(group)),
                "attack_rows": int(group["label"].sum()),
                "attack_rate": float(group["label"].mean()),
                "first_ts": float(group["ts"].min()),
                "last_ts": float(group["ts"].max()),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def build_iot23_detection_latency_table(full_predictions: pd.DataFrame, fractions: list[float]) -> pd.DataFrame:
    rows = []
    for scenario, group in full_predictions.groupby("scenario", sort=False):
        ordered = group.sort_values("ts", kind="mergesort").reset_index(drop=True)
        true_attack_rows = int(ordered["label"].sum())
        for fraction in fractions:
            keep = max(1, int(len(ordered) * fraction))
            prefix = ordered.iloc[:keep]
            rows.append(
                {
                    "scenario": scenario,
                    "fraction": fraction,
                    "rows_seen": int(keep),
                    "scenario_rows_total": int(len(ordered)),
                    "scenario_attack_rows_total": true_attack_rows,
                    "prefix_has_true_attack": bool(prefix["label"].eq(1).any()),
                    "prefix_has_predicted_attack": bool(prefix["y_pred"].eq(1).any()),
                    "prefix_has_true_positive": bool(((prefix["label"] == 1) & (prefix["y_pred"] == 1)).any()),
                }
            )
    return pd.DataFrame(rows)


def first_true_positive_by_scenario(latency_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario, group in latency_df.groupby("scenario", sort=False):
        positive_rows = group[group["prefix_has_true_positive"]]
        first_fraction = None if positive_rows.empty else float(positive_rows.iloc[0]["fraction"])
        rows.append(
            {
                "scenario": scenario,
                "first_true_positive_fraction": first_fraction,
            }
        )
    return pd.DataFrame(rows)


def evaluate_iot23_target_split(
    split_name: str,
    df: pd.DataFrame,
    pipeline: Pipeline,
    feature_cols: list[str],
    categorical_cols: list[str],
    fractions: list[float],
    out_dir: Path,
) -> dict[str, pd.DataFrame]:
    split_dir = out_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    scenario_tables = []

    full_df = prefix_iot23_by_scenario(df, 1.0).copy()
    X_full = normalize_categorical_columns(full_df[feature_cols].copy(), categorical_cols)
    full_df["y_score"] = pipeline.predict_proba(X_full)[:, 1]
    full_df["y_pred"] = pipeline.predict(X_full)

    latency_df = build_iot23_detection_latency_table(full_df, fractions)
    first_tp_df = first_true_positive_by_scenario(latency_df)
    latency_df.to_csv(split_dir / "detection_latency_by_fraction.csv", index=False)
    first_tp_df.to_csv(split_dir / "first_true_positive_fraction.csv", index=False)

    for fraction in fractions:
        prefix_df = prefix_iot23_by_scenario(df, fraction).copy()
        X_eval = normalize_categorical_columns(prefix_df[feature_cols].copy(), categorical_cols)
        prefix_df["y_score"] = pipeline.predict_proba(X_eval)[:, 1]
        prefix_df["y_pred"] = pipeline.predict(X_eval)

        metrics = compute_metrics(prefix_df["label"], prefix_df["y_pred"])
        scenario_summary = summarize_iot23_scenarios(prefix_df)
        merged_first_tp = scenario_summary.merge(first_tp_df, on="scenario", how="left")

        fraction_slug = fraction_to_slug(fraction)
        prefix_df.to_parquet(split_dir / f"predictions_frac_{fraction_slug}.parquet", index=False)
        merged_first_tp.to_csv(split_dir / f"scenario_metrics_frac_{fraction_slug}.csv", index=False)

        malicious_scenarios = latency_df[
            (latency_df["fraction"] == fraction) & (latency_df["scenario_attack_rows_total"] > 0)
        ]
        detection_rate = None
        if not malicious_scenarios.empty:
            detection_rate = float(malicious_scenarios["prefix_has_true_positive"].mean())

        summary_rows.append(
            {
                "split": split_name,
                "fraction": fraction,
                "rows_evaluated": int(len(prefix_df)),
                "n_scenarios": int(prefix_df["scenario"].nunique()),
                "malicious_scenarios_detected_rate": detection_rate,
                **metrics,
            }
        )
        scenario_tables.append(merged_first_tp.assign(split=split_name, fraction=fraction))

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(split_dir / "fraction_summary.csv", index=False)

    scenario_all_df = pd.concat(scenario_tables, ignore_index=True)
    scenario_all_df.to_csv(split_dir / "scenario_metrics_all_fractions.csv", index=False)

    return {
        "summary": summary_df,
        "details": scenario_all_df,
    }


def split_unsw_train_val(df: pd.DataFrame, val_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1.")

    train_parts = []
    val_parts = []

    for label_value, group in df.groupby("label", sort=False):
        n_val = int(round(len(group) * val_fraction))
        if n_val <= 0 or n_val >= len(group):
            raise ValueError(
                f"val_fraction={val_fraction} creates an empty train or validation subset for label={label_value}."
            )

        val_subset = group.sample(n=n_val, random_state=seed)
        train_subset = group.drop(val_subset.index)
        train_parts.append(train_subset)
        val_parts.append(val_subset)

    train_df = (
        pd.concat(train_parts, ignore_index=False)
        .sort_values("id", kind="mergesort")
        .reset_index(drop=True)
    )
    val_df = (
        pd.concat(val_parts, ignore_index=False)
        .sort_values("id", kind="mergesort")
        .reset_index(drop=True)
    )
    return train_df, val_df


def prepare_unsw_eval_frame(df: pd.DataFrame, max_rows: int | None) -> pd.DataFrame:
    ordered = df.sort_values("id", kind="mergesort").reset_index(drop=True)
    if max_rows is None or len(ordered) <= max_rows:
        return ordered
    return ordered.head(max_rows).reset_index(drop=True)


def prefix_unsw_frame(df: pd.DataFrame, fraction: float) -> pd.DataFrame:
    if not 0 < fraction <= 1:
        raise ValueError(f"Invalid fraction: {fraction}")
    keep = max(1, int(len(df) * fraction))
    return df.iloc[:keep].copy().reset_index(drop=True)


def summarize_unsw_attack_categories(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for category, group in pred_df.groupby("attack_cat", sort=False):
        metrics = compute_metrics(group["label"], group["y_pred"])
        rows.append(
            {
                "attack_cat": category,
                "rows": int(len(group)),
                "attack_rows": int(group["label"].sum()),
                "attack_rate": float(group["label"].mean()),
                "first_id": int(group["id"].min()),
                "last_id": int(group["id"].max()),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def first_true_positive_fraction_unsw(predictions_by_fraction: dict[float, pd.DataFrame]) -> float | None:
    for fraction in sorted(predictions_by_fraction):
        pred_df = predictions_by_fraction[fraction]
        if ((pred_df["label"] == 1) & (pred_df["y_pred"] == 1)).any():
            return float(fraction)
    return None


def evaluate_unsw_target_split(
    split_name: str,
    df: pd.DataFrame,
    pipeline: Pipeline,
    feature_cols: list[str],
    categorical_cols: list[str],
    fractions: list[float],
    out_dir: Path,
) -> dict[str, pd.DataFrame]:
    split_dir = out_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    category_tables = []
    predictions_by_fraction: dict[float, pd.DataFrame] = {}

    for fraction in fractions:
        prefix_df = prefix_unsw_frame(df, fraction).copy()
        X_eval = normalize_categorical_columns(prefix_df[feature_cols].copy(), categorical_cols)
        prefix_df["y_score"] = pipeline.predict_proba(X_eval)[:, 1]
        prefix_df["y_pred"] = pipeline.predict(X_eval)
        predictions_by_fraction[fraction] = prefix_df

    first_tp = first_true_positive_fraction_unsw(predictions_by_fraction)
    pd.DataFrame([{"split": split_name, "first_true_positive_fraction": first_tp}]).to_csv(
        split_dir / "first_true_positive_fraction.csv",
        index=False,
    )

    for fraction, pred_df in predictions_by_fraction.items():
        metrics = compute_metrics(pred_df["label"], pred_df["y_pred"])
        category_df = summarize_unsw_attack_categories(pred_df)
        category_df["split"] = split_name
        category_df["fraction"] = fraction
        category_df["first_true_positive_fraction"] = first_tp

        fraction_slug = fraction_to_slug(fraction)
        pred_df.to_parquet(split_dir / f"predictions_frac_{fraction_slug}.parquet", index=False)
        category_df.to_csv(split_dir / f"attack_cat_metrics_frac_{fraction_slug}.csv", index=False)

        summary_rows.append(
            {
                "split": split_name,
                "fraction": fraction,
                "rows_evaluated": int(len(pred_df)),
                "n_attack_categories": int(pred_df["attack_cat"].nunique()),
                "first_true_positive_fraction": first_tp,
                **metrics,
            }
        )
        category_tables.append(category_df)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(split_dir / "fraction_summary.csv", index=False)

    category_all_df = pd.concat(category_tables, ignore_index=True)
    category_all_df.to_csv(split_dir / "attack_cat_metrics_all_fractions.csv", index=False)

    return {
        "summary": summary_df,
        "details": category_all_df,
    }
