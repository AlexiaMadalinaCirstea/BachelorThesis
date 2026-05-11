from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
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
        normalize_categorical_columns,
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
        normalize_categorical_columns,
        prepare_unsw_eval_frame,
        save_json,
        split_unsw_train_val,
        summarize_iot23_scenarios,
        summarize_unsw_attack_categories,
    )


TEMPORAL_PROGRESS_COL = "evidence_progress"
TEMPORAL_PREFIX_COUNT_COL = "prefix_rows_seen"


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


def build_branch_preprocessor(
    categorical_cols: list[str],
    numeric_cols: list[str],
) -> ColumnTransformer:
    transformers: list[tuple[str, Pipeline, list[str]]] = []

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


def transform_frame(
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


def fit_preprocessor(
    df: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
    numeric_cols: list[str],
) -> ColumnTransformer:
    preprocessor = build_branch_preprocessor(categorical_cols, numeric_cols)
    fit_df = normalize_categorical_columns(df[feature_cols].copy(), categorical_cols)
    preprocessor.fit(fit_df)
    return preprocessor


def build_mlp(
    hidden_layers: tuple[int, ...],
    alpha: float,
    batch_size: int,
    max_iter: int,
    seed: int,
) -> MLPClassifier:
    return MLPClassifier(
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


def _build_group_id(df: pd.DataFrame) -> pd.Series:
    if "scenario" in df.columns:
        return df["scenario"].astype("string")
    return pd.Series(["all"] * len(df), index=df.index, dtype="string")


def _build_order_column(df: pd.DataFrame) -> pd.Series:
    if "ts" in df.columns:
        return pd.to_numeric(df["ts"], errors="coerce")
    if "id" in df.columns:
        return pd.to_numeric(df["id"], errors="coerce")
    return pd.Series(np.arange(len(df)), index=df.index, dtype="float64")


def add_temporal_context_features(
    df: pd.DataFrame,
    aligned_numeric_cols: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    context_df = df.copy()
    context_df["_group_id"] = _build_group_id(context_df)
    context_df["_order_col"] = _build_order_column(context_df)
    context_df = context_df.sort_values(["_group_id", "_order_col"], kind="mergesort").reset_index(drop=True)

    group_counts = context_df.groupby("_group_id", sort=False)["_group_id"].transform("count")
    within_index = context_df.groupby("_group_id", sort=False).cumcount() + 1
    context_df[TEMPORAL_PREFIX_COUNT_COL] = within_index.astype(np.int32)
    context_df[TEMPORAL_PROGRESS_COL] = (within_index / group_counts).astype(np.float32)

    temporal_numeric_cols = [TEMPORAL_PREFIX_COUNT_COL, TEMPORAL_PROGRESS_COL]
    for col in aligned_numeric_cols:
        cumulative_col = f"{col}_cummean"
        context_df[cumulative_col] = (
            context_df.groupby("_group_id", sort=False)[col]
            .transform(lambda s: s.expanding().mean())
            .astype(np.float32)
        )
        temporal_numeric_cols.append(cumulative_col)

    context_df = context_df.drop(columns=["_group_id", "_order_col"])
    return downcast_numeric_columns(context_df), temporal_numeric_cols


@dataclass
class MLPBranch:
    name: str
    feature_cols: list[str]
    categorical_cols: list[str]
    numeric_cols: list[str]
    hidden_layers: tuple[int, ...]
    alpha: float
    batch_size: int
    max_iter: int
    seed: int
    preprocessor: ColumnTransformer | None = None
    model: MLPClassifier | None = None

    def fit(self, df: pd.DataFrame, label_col: str = "label") -> None:
        self.preprocessor = fit_preprocessor(df, self.feature_cols, self.categorical_cols, self.numeric_cols)
        X = transform_frame(df, self.preprocessor, self.feature_cols, self.categorical_cols)
        y = df[label_col].to_numpy(dtype=np.int8)
        self.model = build_mlp(self.hidden_layers, self.alpha, self.batch_size, self.max_iter, self.seed)
        self.model.fit(X, y)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.preprocessor is None or self.model is None:
            raise ValueError(f"Branch '{self.name}' is not fitted.")
        X = transform_frame(df, self.preprocessor, self.feature_cols, self.categorical_cols)
        return self.model.predict_proba(X)[:, 1]


@dataclass
class PrototypeBranch:
    feature_cols: list[str]
    categorical_cols: list[str]
    numeric_cols: list[str]
    preprocessor: ColumnTransformer | None = None
    benign_centroid: np.ndarray | None = None
    attack_centroid: np.ndarray | None = None

    def fit(self, df: pd.DataFrame, label_col: str = "label") -> None:
        self.preprocessor = fit_preprocessor(df, self.feature_cols, self.categorical_cols, self.numeric_cols)
        X = transform_frame(df, self.preprocessor, self.feature_cols, self.categorical_cols)
        y = df[label_col].to_numpy(dtype=np.int8)

        benign = X[y == 0]
        attack = X[y == 1]
        if len(benign) == 0 or len(attack) == 0:
            raise ValueError("Prototype branch requires both benign and attack examples.")

        self.benign_centroid = benign.mean(axis=0)
        self.attack_centroid = attack.mean(axis=0)

    def predict_components(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        if self.preprocessor is None or self.benign_centroid is None or self.attack_centroid is None:
            raise ValueError("Prototype branch is not fitted.")

        X = transform_frame(df, self.preprocessor, self.feature_cols, self.categorical_cols)
        benign_dist = np.linalg.norm(X - self.benign_centroid, axis=1)
        attack_dist = np.linalg.norm(X - self.attack_centroid, axis=1)

        logits = np.stack([-benign_dist, -attack_dist], axis=1)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits[:, 1] / exp_logits.sum(axis=1)
        margin = np.abs(benign_dist - attack_dist)

        return {
            "prob_attack": probs.astype(np.float32),
            "benign_distance": benign_dist.astype(np.float32),
            "attack_distance": attack_dist.astype(np.float32),
            "prototype_margin": margin.astype(np.float32),
        }


@dataclass
class InterpretableHybridPredictor:
    tabular_branch: MLPBranch
    temporal_branch: MLPBranch
    prototype_branch: PrototypeBranch
    threshold: float = 0.5

    def _confidence(self, probs: np.ndarray) -> np.ndarray:
        return np.clip(np.abs(probs - 0.5) * 2.0, 0.0, 1.0)

    def predict_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        tab_prob = self.tabular_branch.predict_proba(df)
        temp_prob = self.temporal_branch.predict_proba(df)
        proto_components = self.prototype_branch.predict_components(df)
        proto_prob = proto_components["prob_attack"]

        evidence_progress = np.clip(df[TEMPORAL_PROGRESS_COL].to_numpy(dtype=np.float32), 0.0, 1.0)

        tab_conf = self._confidence(tab_prob)
        temp_conf = self._confidence(temp_prob)
        proto_conf = self._confidence(proto_prob)
        proto_margin = proto_components["prototype_margin"]
        proto_margin_scaled = proto_margin / (proto_margin + 1.0)

        branch_attack_votes = (
            (tab_prob >= self.threshold).astype(np.int8)
            + (temp_prob >= self.threshold).astype(np.int8)
            + (proto_prob >= self.threshold).astype(np.int8)
        )
        branch_agreement = np.where(
            (branch_attack_votes == 0) | (branch_attack_votes == 3),
            1.0,
            np.where(branch_attack_votes == 1, 1.0 / 3.0, 2.0 / 3.0),
        ).astype(np.float32)

        temporal_prior = 1.05 - 0.70 * evidence_progress
        tabular_prior = 0.55 + 0.70 * evidence_progress
        prototype_prior = 0.60 + 0.40 * (1.0 - np.abs(tab_prob - temp_prob))

        prototype_boost = 1.0 + 0.35 * proto_margin_scaled + 0.25 * (1.0 - branch_agreement)

        w_tab = np.clip(tabular_prior * (0.45 + tab_conf), 1e-6, None)
        w_temp = np.clip(temporal_prior * (0.45 + temp_conf), 1e-6, None)
        w_proto = np.clip(prototype_prior * prototype_boost * (0.40 + proto_conf), 1e-6, None)

        weight_sum = w_tab + w_temp + w_proto
        w_tab = w_tab / weight_sum
        w_temp = w_temp / weight_sum
        w_proto = w_proto / weight_sum

        final_prob = (w_tab * tab_prob) + (w_temp * temp_prob) + (w_proto * proto_prob)
        final_pred = (final_prob >= self.threshold).astype(np.int8)

        return pd.DataFrame(
            {
                "tabular_prob": tab_prob.astype(np.float32),
                "temporal_prob": temp_prob.astype(np.float32),
                "prototype_prob": proto_prob.astype(np.float32),
                "tabular_confidence": tab_conf.astype(np.float32),
                "temporal_confidence": temp_conf.astype(np.float32),
                "prototype_confidence": proto_conf.astype(np.float32),
                "prototype_margin": proto_margin.astype(np.float32),
                "branch_agreement": branch_agreement.astype(np.float32),
                "tabular_weight": w_tab.astype(np.float32),
                "temporal_weight": w_temp.astype(np.float32),
                "prototype_weight": w_proto.astype(np.float32),
                "fused_prob": final_prob.astype(np.float32),
                "y_pred": final_pred.astype(np.int8),
            }
        )


def train_interpretable_hybrid(
    source_df: pd.DataFrame,
    aligned_feature_cols: list[str],
    aligned_categorical_cols: list[str],
    aligned_numeric_cols: list[str],
    temporal_feature_cols: list[str],
    seed: int,
    tabular_hidden_layers: tuple[int, ...],
    temporal_hidden_layers: tuple[int, ...],
    alpha: float,
    batch_size: int,
    max_iter: int,
) -> InterpretableHybridPredictor:
    tabular_branch = MLPBranch(
        name="tabular",
        feature_cols=aligned_feature_cols,
        categorical_cols=aligned_categorical_cols,
        numeric_cols=aligned_numeric_cols,
        hidden_layers=tabular_hidden_layers,
        alpha=alpha,
        batch_size=batch_size,
        max_iter=max_iter,
        seed=seed,
    )
    tabular_branch.fit(source_df)

    temporal_branch = MLPBranch(
        name="temporal",
        feature_cols=temporal_feature_cols,
        categorical_cols=aligned_categorical_cols,
        numeric_cols=[col for col in temporal_feature_cols if col not in aligned_categorical_cols],
        hidden_layers=temporal_hidden_layers,
        alpha=alpha,
        batch_size=batch_size,
        max_iter=max_iter,
        seed=seed + 17,
    )
    temporal_branch.fit(source_df)

    prototype_branch = PrototypeBranch(
        feature_cols=aligned_feature_cols,
        categorical_cols=aligned_categorical_cols,
        numeric_cols=aligned_numeric_cols,
    )
    prototype_branch.fit(source_df)

    return InterpretableHybridPredictor(
        tabular_branch=tabular_branch,
        temporal_branch=temporal_branch,
        prototype_branch=prototype_branch,
    )


def _iot23_first_true_positive(pred_df: pd.DataFrame, fractions: list[float]) -> pd.DataFrame:
    rows = []
    for scenario, group in pred_df.groupby("scenario", sort=False):
        ordered = group.sort_values("ts", kind="mergesort").reset_index(drop=True)
        first_fraction = None
        for fraction in fractions:
            keep = max(1, int(len(ordered) * fraction))
            prefix = ordered.iloc[:keep]
            if ((prefix["label"] == 1) & (prefix["y_pred"] == 1)).any():
                first_fraction = float(fraction)
                break
        rows.append({"scenario": scenario, "first_true_positive_fraction": first_fraction})
    return pd.DataFrame(rows)


def evaluate_hybrid_iot23_target_split(
    split_name: str,
    df: pd.DataFrame,
    predictor: InterpretableHybridPredictor,
    fractions: list[float],
    out_dir: Path,
) -> dict[str, pd.DataFrame]:
    split_dir = out_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    detail_rows = []

    first_tp_df = _iot23_first_true_positive(df, fractions)
    first_tp_df.to_csv(split_dir / "first_true_positive_fraction.csv", index=False)

    for fraction in fractions:
        parts = []
        for _, group in df.groupby("scenario", sort=False):
            keep = max(1, int(len(group) * fraction))
            parts.append(group.iloc[:keep])
        prefix_df = pd.concat(parts, ignore_index=True).copy()

        branch_df = predictor.predict_frame(prefix_df)
        pred_df = pd.concat([prefix_df.reset_index(drop=True), branch_df], axis=1)
        pred_df["y_score"] = pred_df["fused_prob"]

        metrics = compute_metrics(pred_df["label"], pred_df["y_pred"])
        scenario_df = summarize_iot23_scenarios(pred_df).merge(first_tp_df, on="scenario", how="left")
        scenario_df["split"] = split_name
        scenario_df["fraction"] = fraction

        fraction_slug = fraction_to_slug(fraction)
        pred_df.to_parquet(split_dir / f"predictions_frac_{fraction_slug}.parquet", index=False)
        scenario_df.to_csv(split_dir / f"scenario_metrics_frac_{fraction_slug}.csv", index=False)

        summary_rows.append(
            {
                "split": split_name,
                "fraction": fraction,
                "rows_evaluated": int(len(pred_df)),
                "n_scenarios": int(pred_df["scenario"].nunique()),
                "mean_tabular_weight": float(pred_df["tabular_weight"].mean()),
                "mean_temporal_weight": float(pred_df["temporal_weight"].mean()),
                "mean_prototype_weight": float(pred_df["prototype_weight"].mean()),
                **metrics,
            }
        )
        detail_rows.append(scenario_df)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(split_dir / "fraction_summary.csv", index=False)

    detail_df = pd.concat(detail_rows, ignore_index=True)
    detail_df.to_csv(split_dir / "scenario_metrics_all_fractions.csv", index=False)
    return {"summary": summary_df, "details": detail_df}


def _unsw_first_true_positive(df: pd.DataFrame, fractions: list[float]) -> float | None:
    for fraction in fractions:
        keep = max(1, int(len(df) * fraction))
        prefix = df.iloc[:keep]
        if ((prefix["label"] == 1) & (prefix["y_pred"] == 1)).any():
            return float(fraction)
    return None


def evaluate_hybrid_unsw_target_split(
    split_name: str,
    df: pd.DataFrame,
    predictor: InterpretableHybridPredictor,
    fractions: list[float],
    out_dir: Path,
) -> dict[str, pd.DataFrame]:
    split_dir = out_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    detail_rows = []

    full_pred_df = pd.concat([df.reset_index(drop=True), predictor.predict_frame(df)], axis=1)
    first_tp = _unsw_first_true_positive(full_pred_df, fractions)
    pd.DataFrame([{"split": split_name, "first_true_positive_fraction": first_tp}]).to_csv(
        split_dir / "first_true_positive_fraction.csv",
        index=False,
    )

    for fraction in fractions:
        keep = max(1, int(len(full_pred_df) * fraction))
        pred_df = full_pred_df.iloc[:keep].copy().reset_index(drop=True)
        pred_df["y_score"] = pred_df["fused_prob"]

        metrics = compute_metrics(pred_df["label"], pred_df["y_pred"])
        attack_cat_df = summarize_unsw_attack_categories(pred_df)
        attack_cat_df["split"] = split_name
        attack_cat_df["fraction"] = fraction
        attack_cat_df["first_true_positive_fraction"] = first_tp

        fraction_slug = fraction_to_slug(fraction)
        pred_df.to_parquet(split_dir / f"predictions_frac_{fraction_slug}.parquet", index=False)
        attack_cat_df.to_csv(split_dir / f"attack_cat_metrics_frac_{fraction_slug}.csv", index=False)

        summary_rows.append(
            {
                "split": split_name,
                "fraction": fraction,
                "rows_evaluated": int(len(pred_df)),
                "n_attack_categories": int(pred_df["attack_cat"].nunique()),
                "first_true_positive_fraction": first_tp,
                "mean_tabular_weight": float(pred_df["tabular_weight"].mean()),
                "mean_temporal_weight": float(pred_df["temporal_weight"].mean()),
                "mean_prototype_weight": float(pred_df["prototype_weight"].mean()),
                **metrics,
            }
        )
        detail_rows.append(attack_cat_df)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(split_dir / "fraction_summary.csv", index=False)

    detail_df = pd.concat(detail_rows, ignore_index=True)
    detail_df.to_csv(split_dir / "attack_cat_metrics_all_fractions.csv", index=False)
    return {"summary": summary_df, "details": detail_df}
