from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from evaluate import compute_overall_metrics_binary
from train_baseline_rf import NUMERIC_COLS, CATEGORICAL_COLS, validate_columns

FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS


def build_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="constant", fill_value=0.0))]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_COLS),
            ("cat", categorical_transformer, CATEGORICAL_COLS),
        ]
    )


def build_xgb_pipeline(seed: int, scale_pos_weight: float, params: dict[str, Any]) -> Pipeline:
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        **params,
    )

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )


def make_param_grid() -> List[dict[str, Any]]:
    return [
        {
            "n_estimators": 250,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "reg_lambda": 1.0,
        },
        {
            "n_estimators": 400,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "reg_lambda": 5.0,
        },
        {
            "n_estimators": 350,
            "max_depth": 5,
            "learning_rate": 0.03,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_weight": 3,
            "reg_lambda": 5.0,
        },
        {
            "n_estimators": 500,
            "max_depth": 5,
            "learning_rate": 0.03,
            "subsample": 0.7,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "reg_lambda": 10.0,
        },
        {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 1.0,
            "min_child_weight": 1,
            "reg_lambda": 1.0,
        },
        {
            "n_estimators": 450,
            "max_depth": 6,
            "learning_rate": 0.03,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_weight": 5,
            "reg_lambda": 10.0,
        },
    ]


def compute_scale_pos_weight(y: pd.Series) -> float:
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0:
        return 1.0
    return max(n_neg / n_pos, 1e-6)


def pick_inner_split(train_df: pd.DataFrame, seed: int, val_frac: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    scenarios = train_df["scenario"].astype(str)
    unique_scenarios = scenarios.nunique()
    if unique_scenarios < 3:
        raise AssertionError(
            "Need at least 3 training scenarios to create an inner scenario-aware validation split"
        )

    gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    train_idx, val_idx = next(gss.split(train_df, groups=scenarios))
    inner_train = train_df.iloc[train_idx].copy()
    inner_val = train_df.iloc[val_idx].copy()

    if inner_train.empty or inner_val.empty:
        raise AssertionError("Inner split produced an empty subset")

    return inner_train, inner_val


def macro_f1_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> float:
    y_pred = (y_score >= threshold).astype(int)
    return float(
        precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)[2]
    )


def find_best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    thresholds = np.linspace(0.10, 0.90, 33)
    best_threshold = 0.5
    best_score = -1.0

    for thr in thresholds:
        score = macro_f1_at_threshold(y_true, y_score, thr)
        if score > best_score:
            best_score = score
            best_threshold = float(thr)

    return best_threshold, best_score


def tune_on_inner_split(
    inner_train: pd.DataFrame,
    inner_val: pd.DataFrame,
    target_col: str,
    seed: int,
    tune_threshold: bool,
) -> dict[str, Any]:
    validate_columns(inner_train, target_col)
    validate_columns(inner_val, target_col)

    X_train = inner_train[FEATURE_COLS].copy()
    y_train = inner_train[target_col].copy()
    X_val = inner_val[FEATURE_COLS].copy()
    y_val = inner_val[target_col].copy()

    scale_pos_weight = compute_scale_pos_weight(y_train)
    candidates = []

    for i, params in enumerate(make_param_grid(), start=1):
        pipeline = build_xgb_pipeline(seed=seed, scale_pos_weight=scale_pos_weight, params=params)
        pipeline.fit(X_train, y_train)

        val_score = pipeline.predict_proba(X_val)[:, 1]
        if tune_threshold:
            threshold, tuned_macro_f1 = find_best_threshold(y_val.to_numpy(), val_score)
        else:
            threshold = 0.5
            tuned_macro_f1 = macro_f1_at_threshold(y_val.to_numpy(), val_score, threshold)

        val_pred = (val_score >= threshold).astype(int)
        metrics = compute_overall_metrics_binary(
            y_true=y_val.to_numpy(),
            y_pred=val_pred,
            y_score=val_score,
            n_boot=200,
            seed=seed,
        )

        candidates.append(
            {
                "candidate_id": i,
                "params": params,
                "scale_pos_weight": float(scale_pos_weight),
                "threshold": float(threshold),
                "inner_val_f1_macro": float(metrics["f1_macro"]),
                "inner_val_balanced_accuracy": float(metrics["balanced_accuracy"]),
                "inner_val_recall_attack": float(metrics["per_class"]["1"]["recall"]),
                "inner_val_recall_benign": float(metrics["per_class"]["0"]["recall"]),
                "inner_val_pr_auc": metrics.get("pr_auc"),
                "inner_val_roc_auc": metrics.get("roc_auc"),
                "tuned_macro_f1": float(tuned_macro_f1),
            }
        )

    candidates_df = pd.DataFrame(candidates).sort_values(
        ["inner_val_f1_macro", "inner_val_balanced_accuracy", "inner_val_recall_attack"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    best = candidates_df.iloc[0].to_dict()
    return {
        "best": best,
        "candidates_df": candidates_df,
    }

