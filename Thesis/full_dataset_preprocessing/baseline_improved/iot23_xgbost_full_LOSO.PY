from __future__ import annotations

import argparse
import gc
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from baseline_common import (
    CATEGORICAL_COLS,
    FEATURE_COLS,
    NUMERIC_COLS,
    save_json,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def compute_scale_pos_weight(y: pd.Series) -> float:
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0:
        return 1.0
    return max(n_neg / n_pos, 1e-6)


def build_pipeline(
    seed: int,
    n_jobs: int,
    scale_pos_weight: float,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
) -> Pipeline:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_COLS),
        ("cat", categorical_transformer, CATEGORICAL_COLS),
    ])
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=n_jobs,
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_lambda=5.0,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den != 0 else 0.0


def _f1(precision: float, recall: float) -> float:
    den = precision + recall
    return float(2 * precision * recall / den) if den != 0 else 0.0


def _metrics_from_confusion(cm: np.ndarray) -> dict:
    tn, fp = int(cm[0, 0]), int(cm[0, 1])
    fn, tp = int(cm[1, 0]), int(cm[1, 1])

    support_0 = tn + fp
    support_1 = fn + tp
    total = support_0 + support_1

    precision_0 = _safe_div(tn, tn + fn)
    recall_0 = _safe_div(tn, tn + fp)
    f1_0 = _f1(precision_0, recall_0)

    precision_1 = _safe_div(tp, tp + fp)
    recall_1 = _safe_div(tp, tp + fn)
    f1_1 = _f1(precision_1, recall_1)

    accuracy = _safe_div(tn + tp, total)
    balanced_accuracy = (recall_0 + recall_1) / 2.0
    precision_macro = (precision_0 + precision_1) / 2.0
    recall_macro = (recall_0 + recall_1) / 2.0
    f1_macro = (f1_0 + f1_1) / 2.0

    precision_weighted = _safe_div(
        precision_0 * support_0 + precision_1 * support_1,
        total,
    )
    recall_weighted = _safe_div(
        recall_0 * support_0 + recall_1 * support_1,
        total,
    )
    f1_weighted = _safe_div(
        f1_0 * support_0 + f1_1 * support_1,
        total,
    )

    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
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
        "per_class": {
            "0": {
                "precision": float(precision_0),
                "recall": float(recall_0),
                "f1": float(f1_0),
                "support": int(support_0),
            },
            "1": {
                "precision": float(precision_1),
                "recall": float(recall_1),
                "f1": float(f1_1),
                "support": int(support_1),
            },
        },
        "n_rows": int(total),
        "recall_attack": float(recall_1),
        "recall_benign": float(recall_0),
    }


def extract_scenario_name(path: Path) -> str:
    return path.stem


def read_scenario_df(path: Path, target_col: str) -> pd.DataFrame:
    required_columns = [target_col] + FEATURE_COLS
    df = pd.read_parquet(path, columns=required_columns)
    df["scenario"] = extract_scenario_name(path)
    return df


def sample_single_scenario_df(
    path: Path,
    target_col: str,
    sample_frac: float,
    seed: int,
) -> pd.DataFrame:
    df = read_scenario_df(path, target_col)

    if df.empty:
        return df

    if sample_frac >= 1.0:
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    rng = np.random.RandomState(seed)
    sampled_parts = []

    for _, group in df.groupby(target_col):
        n_take = max(1, int(round(len(group) * sample_frac)))
        n_take = min(n_take, len(group))
        idx = rng.choice(group.index.to_numpy(), size=n_take, replace=False)
        sampled_parts.append(group.loc[idx])

    sampled = pd.concat(sampled_parts, ignore_index=True)
    sampled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    del df, sampled_parts
    gc.collect()
    return sampled


def cap_sampled_train_df(
    df: pd.DataFrame,
    target_col: str,
    max_rows: int | None,
    seed: int,
) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df

    rng = np.random.RandomState(seed)
    capped_parts = []

    for _, group in df.groupby(target_col):
        frac = len(group) / len(df)
        class_cap = max(1, int(round(max_rows * frac)))
        n_take = min(class_cap, len(group))
        idx = rng.choice(group.index.to_numpy(), size=n_take, replace=False)
        capped_parts.append(group.loc[idx])

    capped = pd.concat(capped_parts, ignore_index=True)
    capped = capped.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    del capped_parts, df
    gc.collect()
    return capped


def load_train_df_for_fold(
    scenario_paths: list[Path],
    held_out_path: Path,
    target_col: str,
    sample_frac: float,
    max_rows: int | None,
    seed: int,
) -> pd.DataFrame:
    sampled_dfs = []
    train_candidates = [p for p in scenario_paths if p != held_out_path]

    for i, path in enumerate(train_candidates, start=1):
        log.info(
            "Loading + sampling training scenario %d/%d: %s",
            i,
            len(train_candidates),
            path.stem,
        )
        sampled_df = sample_single_scenario_df(
            path=path,
            target_col=target_col,
            sample_frac=sample_frac,
            seed=seed,
        )

        if not sampled_df.empty:
            sampled_dfs.append(sampled_df)

        gc.collect()

    if not sampled_dfs:
        raise RuntimeError("No sampled training data produced for this fold.")

    train_df = pd.concat(sampled_dfs, ignore_index=True)
    train_df = cap_sampled_train_df(
        df=train_df,
        target_col=target_col,
        max_rows=max_rows,
        seed=seed,
    )

    del sampled_dfs
    gc.collect()
    return train_df


def evaluate_one_scenario(
    pipeline: Pipeline,
    scenario_path: Path,
    out_path: Path,
    target_col: str,
    decision_threshold: float,
    batch_size: int,
) -> dict:
    required_columns = [target_col] + FEATURE_COLS
    parquet_file = pq.ParquetFile(scenario_path)

    writer = None
    cm = np.zeros((2, 2), dtype=np.int64)
    chunk_count = 0
    scenario_name = extract_scenario_name(scenario_path)

    try:
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=required_columns):
            chunk_count += 1
            df = batch.to_pandas()
            df["scenario"] = scenario_name

            y_true = df[target_col].to_numpy(dtype=np.int8, copy=False)
            X = df[FEATURE_COLS]

            y_score = pipeline.predict_proba(X)[:, 1]
            y_pred = (y_score >= decision_threshold).astype(np.int8)

            cm[0, 0] += int(((y_true == 0) & (y_pred == 0)).sum())
            cm[0, 1] += int(((y_true == 0) & (y_pred == 1)).sum())
            cm[1, 0] += int(((y_true == 1) & (y_pred == 0)).sum())
            cm[1, 1] += int(((y_true == 1) & (y_pred == 1)).sum())

            pred_df = pd.DataFrame({
                "scenario": df["scenario"].astype(str),
                "label_binary": y_true.astype(np.int8),
                "y_pred": y_pred.astype(np.int8),
                "y_score": y_score.astype(np.float32),
            })

            table = pa.Table.from_pandas(pred_df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema)
            writer.write_table(table)

            del df, X, y_true, y_score, y_pred, pred_df, table, batch
            gc.collect()

        log.info("Evaluated %s in %d batch(es)", scenario_name, chunk_count)

    finally:
        if writer is not None:
            writer.close()

    return _metrics_from_confusion(cm)


def aggregate_fold_statistics(folds: list[dict]) -> tuple[dict, dict]:
    metric_keys = [
        "accuracy",
        "balanced_accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
        "recall_attack",
        "recall_benign",
    ]

    fold_mean = {}
    fold_std = {}

    for key in metric_keys:
        values = [fold["metrics"][key] for fold in folds]
        fold_mean[key] = float(np.mean(values))
        fold_std[key] = float(np.std(values, ddof=0))

    return fold_mean, fold_std


def main() -> None:
    parser = argparse.ArgumentParser(description="LOSO XGBoost on IoT-23 scenario parquet files")
    parser.add_argument("--scenario_dir", required=True, help="Directory containing one parquet per scenario")
    parser.add_argument("--out_dir", required=True, help="Directory to save LOSO outputs")
    parser.add_argument("--target_col", default="label_binary")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_sample_frac", type=float, default=0.10)
    parser.add_argument("--max_train_rows", type=int, default=3_000_000)
    parser.add_argument("--batch_size", type=int, default=250_000)
    parser.add_argument("--n_estimators", type=int, default=250)
    parser.add_argument("--max_depth", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--n_jobs", type=int, default=4)
    parser.add_argument("--decision_threshold", type=float, default=0.50)
    args = parser.parse_args()

    if not (0.0 < args.decision_threshold < 1.0):
        raise ValueError("--decision_threshold must be between 0 and 1")

    scenario_dir = Path(args.scenario_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenario_paths = sorted(scenario_dir.glob("*.parquet"))
    if not scenario_paths:
        raise FileNotFoundError(f"No parquet files found in {scenario_dir}")

    log.info("Found %d scenario parquet files", len(scenario_paths))

    folds = []
    pooled_cm = np.zeros((2, 2), dtype=np.int64)

    for fold_idx, held_out_path in enumerate(scenario_paths, start=1):
        held_out_name = extract_scenario_name(held_out_path)
        log.info("Fold %d/%d | held-out scenario: %s", fold_idx, len(scenario_paths), held_out_name)

        train_df = load_train_df_for_fold(
            scenario_paths=scenario_paths,
            held_out_path=held_out_path,
            target_col=args.target_col,
            sample_frac=args.train_sample_frac,
            max_rows=args.max_train_rows,
            seed=args.seed,
        )

        sampled_train_rows = int(len(train_df))
        X_train = train_df[FEATURE_COLS]
        y_train = train_df[args.target_col]
        scale_pos_weight = compute_scale_pos_weight(y_train)

        pipeline = build_pipeline(
            seed=args.seed,
            n_jobs=args.n_jobs,
            scale_pos_weight=scale_pos_weight,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
        )

        log.info("Training fold model for held-out scenario %s ...", held_out_name)
        pipeline.fit(X_train, y_train)

        fold_dir = out_dir / held_out_name
        fold_dir.mkdir(parents=True, exist_ok=True)

        model_path = fold_dir / "xgb_model.joblib"
        pred_path = fold_dir / "predictions.parquet"
        joblib.dump(pipeline, model_path)

        metrics = evaluate_one_scenario(
            pipeline=pipeline,
            scenario_path=held_out_path,
            out_path=pred_path,
            target_col=args.target_col,
            decision_threshold=args.decision_threshold,
            batch_size=args.batch_size,
        )

        cm = np.array(metrics["confusion_matrix"]["matrix"], dtype=np.int64)
        pooled_cm += cm

        fold_summary = {
            "scenario": held_out_name,
            "train_rows_sampled": sampled_train_rows,
            "test_rows": int(metrics["n_rows"]),
            "scale_pos_weight": float(scale_pos_weight),
            "metrics": metrics,
            "outputs": {
                "model_path": str(model_path),
                "predictions_path": str(pred_path),
            },
        }
        save_json(fold_summary, fold_dir / "fold_summary.json")
        folds.append(fold_summary)

        del train_df, X_train, y_train, pipeline
        gc.collect()

    pooled_metrics = _metrics_from_confusion(pooled_cm)
    fold_mean, fold_std = aggregate_fold_statistics(folds)

    worst_fold = min(folds, key=lambda x: x["metrics"]["f1_macro"])
    best_fold = max(folds, key=lambda x: x["metrics"]["f1_macro"])

    loso_summary = {
        "model": "XGBClassifier",
        "protocol": "LOSO",
        "target_col": args.target_col,
        "seed": int(args.seed),
        "decision_threshold": float(args.decision_threshold),
        "train_sample_frac": float(args.train_sample_frac),
        "max_train_rows": int(args.max_train_rows) if args.max_train_rows is not None else None,
        "batch_size": int(args.batch_size),
        "n_estimators": int(args.n_estimators),
        "max_depth": int(args.max_depth),
        "learning_rate": float(args.learning_rate),
        "n_jobs": int(args.n_jobs),
        "n_folds": int(len(folds)),
        "scenario_dir": str(scenario_dir),
        "pooled_overall": pooled_metrics,
        "fold_mean": fold_mean,
        "fold_std": fold_std,
        "worst_fold_by_f1_macro": {
            "scenario": worst_fold["scenario"],
            **worst_fold["metrics"],
        },
        "best_fold_by_f1_macro": {
            "scenario": best_fold["scenario"],
            **best_fold["metrics"],
        },
        "folds": folds,
        "notes": [
            "Each fold holds out exactly one scenario parquet as test data.",
            "Training is performed on sampled rows from all remaining scenarios only.",
            "Sampling is done per scenario before concatenation to reduce memory usage.",
            "Held-out scenario evaluation is performed in batches to avoid memory blow-up.",
            "This implements scenario-level Leave-One-Scenario-Out evaluation.",
            "Decision threshold is applied after probability prediction.",
        ],
    }

    save_json(loso_summary, out_dir / "loso_summary.json")
    log.info("Saved LOSO summary to %s", out_dir / "loso_summary.json")
    log.info("Pooled overall metrics: %s", pooled_metrics)
    log.info("Fold mean metrics: %s", fold_mean)


if __name__ == "__main__":
    main()