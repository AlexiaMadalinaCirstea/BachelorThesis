from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
import joblib

from evaluate import compute_overall_metrics_binary
from splitters import generate_loso_splits
from train_baseline_xgboost import (
    FEATURE_COLS,
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    build_xgb_pipeline,
    compute_scale_pos_weight,
    make_param_grid,
    pick_inner_split,
    tune_on_inner_split,
    validate_columns,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def save_json(obj: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def load_all_flows(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    df = pd.read_parquet(path)
    if df.empty:
        raise AssertionError("all_flows.parquet is empty")
    return df


def validate_all_flows(df: pd.DataFrame, target_col: str) -> None:
    validate_columns(df, target_col)
    if "scenario" not in df.columns:
        raise AssertionError("Missing required column: scenario")
    if df["scenario"].isna().any():
        raise AssertionError("Found missing scenario values")
    if target_col not in df.columns:
        raise AssertionError(f"Missing target column: {target_col}")


def make_predictions_df(test_df: pd.DataFrame, y_pred, y_score, target_col: str) -> pd.DataFrame:
    out = test_df[["scenario", target_col]].copy()
    out = out.rename(columns={target_col: "label_binary"})
    out["y_pred"] = y_pred
    out["y_score"] = y_score
    return out


def compute_fold_summary(test_df: pd.DataFrame, pred_df: pd.DataFrame) -> Dict[str, object]:
    y_true = pred_df["label_binary"]
    y_pred = pred_df["y_pred"]
    y_score = pred_df["y_score"]

    overall = compute_overall_metrics_binary(
        y_true=y_true.to_numpy(),
        y_pred=y_pred.to_numpy(),
        y_score=y_score.to_numpy(),
        n_boot=1000,
        seed=42,
    )

    return {
        "scenario": str(test_df["scenario"].iloc[0]),
        "n_rows": int(len(test_df)),
        "n_benign": int((test_df["label_binary"] == 0).sum()),
        "n_attack": int((test_df["label_binary"] == 1).sum()),
        "accuracy": float(overall["accuracy"]),
        "balanced_accuracy": float(overall["balanced_accuracy"]),
        "f1_macro": float(overall["f1_macro"]),
        "f1_weighted": float(overall["f1_weighted"]),
        "precision_macro": float(overall["precision_macro"]),
        "recall_macro": float(overall["recall_macro"]),
        "roc_auc": overall.get("roc_auc"),
        "pr_auc": overall.get("pr_auc"),
        "precision_benign": float(overall["per_class"]["0"]["precision"]),
        "recall_benign": float(overall["per_class"]["0"]["recall"]),
        "f1_benign": float(overall["per_class"]["0"]["f1"]),
        "support_benign": int(overall["per_class"]["0"]["support"]),
        "precision_attack": float(overall["per_class"]["1"]["precision"]),
        "recall_attack": float(overall["per_class"]["1"]["recall"]),
        "f1_attack": float(overall["per_class"]["1"]["f1"]),
        "support_attack": int(overall["per_class"]["1"]["support"]),
    }


def compute_aggregate_summary(all_preds: pd.DataFrame, folds_df: pd.DataFrame) -> Dict[str, object]:
    y_true = all_preds["label_binary"]
    y_pred = all_preds["y_pred"]
    y_score = all_preds["y_score"]

    pooled = compute_overall_metrics_binary(
        y_true=y_true.to_numpy(),
        y_pred=y_pred.to_numpy(),
        y_score=y_score.to_numpy(),
        n_boot=1000,
        seed=42,
    )

    return {
        "n_rows_total": int(len(all_preds)),
        "n_scenarios": int(folds_df["scenario"].nunique()),
        "pooled_overall": pooled,
        "fold_mean": {
            "accuracy": float(folds_df["accuracy"].mean()),
            "balanced_accuracy": float(folds_df["balanced_accuracy"].mean()),
            "f1_macro": float(folds_df["f1_macro"].mean()),
            "f1_weighted": float(folds_df["f1_weighted"].mean()),
            "precision_macro": float(folds_df["precision_macro"].mean()),
            "recall_macro": float(folds_df["recall_macro"].mean()),
            "recall_attack": float(folds_df["recall_attack"].mean()),
            "recall_benign": float(folds_df["recall_benign"].mean()),
        },
        "fold_std": {
            "accuracy": float(folds_df["accuracy"].std(ddof=1)) if len(folds_df) > 1 else 0.0,
            "balanced_accuracy": float(folds_df["balanced_accuracy"].std(ddof=1)) if len(folds_df) > 1 else 0.0,
            "f1_macro": float(folds_df["f1_macro"].std(ddof=1)) if len(folds_df) > 1 else 0.0,
            "f1_weighted": float(folds_df["f1_weighted"].std(ddof=1)) if len(folds_df) > 1 else 0.0,
            "precision_macro": float(folds_df["precision_macro"].std(ddof=1)) if len(folds_df) > 1 else 0.0,
            "recall_macro": float(folds_df["recall_macro"].std(ddof=1)) if len(folds_df) > 1 else 0.0,
            "recall_attack": float(folds_df["recall_attack"].std(ddof=1)) if len(folds_df) > 1 else 0.0,
            "recall_benign": float(folds_df["recall_benign"].std(ddof=1)) if len(folds_df) > 1 else 0.0,
        },
        "worst_fold_by_f1_macro": folds_df.sort_values("f1_macro", ascending=True).iloc[0].to_dict(),
        "best_fold_by_f1_macro": folds_df.sort_values("f1_macro", ascending=False).iloc[0].to_dict(),
    }


def run_fold(
    df: pd.DataFrame,
    split: Dict[str, List[str]],
    out_dir: Path,
    target_col: str,
    seed: int,
    val_frac: float,
    tune_threshold: bool,
) -> Dict[str, object]:
    fold_name = split["name"]
    train_scenarios = split["train"]
    test_scenario = split["test"][0]

    fold_dir = out_dir / fold_name
    fold_dir.mkdir(parents=True, exist_ok=True)

    outer_train_df = df[df["scenario"].isin(train_scenarios)].copy()
    test_df = df[df["scenario"] == test_scenario].copy()

    if outer_train_df.empty:
        raise AssertionError(f"{fold_name}: outer_train_df is empty")
    if test_df.empty:
        raise AssertionError(f"{fold_name}: test_df is empty")

    inner_train_df, inner_val_df = pick_inner_split(outer_train_df, seed=seed, val_frac=val_frac)

    log.info(
        "[%s] outer train scenarios=%d rows=%d | test scenario=%s rows=%d",
        fold_name,
        len(train_scenarios),
        len(outer_train_df),
        test_scenario,
        len(test_df),
    )
    log.info(
        "[%s] inner train scenarios=%d rows=%d | inner val scenarios=%d rows=%d",
        fold_name,
        inner_train_df["scenario"].nunique(),
        len(inner_train_df),
        inner_val_df["scenario"].nunique(),
        len(inner_val_df),
    )

    tuning = tune_on_inner_split(
        inner_train=inner_train_df,
        inner_val=inner_val_df,
        target_col=target_col,
        seed=seed,
        tune_threshold=tune_threshold,
    )
    best = tuning["best"]
    best_params = dict(best["params"])
    best_threshold = float(best["threshold"])
    outer_scale_pos_weight = compute_scale_pos_weight(outer_train_df[target_col])

    tuning["candidates_df"].to_csv(fold_dir / "inner_tuning_results.csv", index=False)

    X_outer_train = outer_train_df[FEATURE_COLS].copy()
    y_outer_train = outer_train_df[target_col].copy()
    X_test = test_df[FEATURE_COLS].copy()

    final_pipeline = build_xgb_pipeline(
        seed=seed,
        scale_pos_weight=outer_scale_pos_weight,
        params=best_params,
    )
    final_pipeline.fit(X_outer_train, y_outer_train)

    joblib.dump(final_pipeline, fold_dir / "model.joblib")

    test_score = final_pipeline.predict_proba(X_test)[:, 1]
    test_pred = (test_score >= best_threshold).astype(int)

    pred_df = make_predictions_df(test_df=test_df, y_pred=test_pred, y_score=test_score, target_col=target_col)
    pred_path = fold_dir / "xgb_test_predictions.parquet"
    pred_df.to_parquet(pred_path, index=False)

    fold_summary = compute_fold_summary(test_df=test_df, pred_df=pred_df)
    fold_summary["fold_name"] = fold_name
    fold_summary["predictions_path"] = str(pred_path)
    fold_summary["train_scenarios"] = train_scenarios
    fold_summary["test_scenarios"] = [test_scenario]
    fold_summary["inner_train_scenarios"] = sorted(inner_train_df["scenario"].astype(str).unique().tolist())
    fold_summary["inner_val_scenarios"] = sorted(inner_val_df["scenario"].astype(str).unique().tolist())
    fold_summary["best_params"] = best_params
    fold_summary["best_threshold"] = best_threshold
    fold_summary["inner_scale_pos_weight"] = float(best["scale_pos_weight"])
    fold_summary["outer_scale_pos_weight"] = float(outer_scale_pos_weight)
    fold_summary["inner_val_metrics"] = {
        "f1_macro": float(best["inner_val_f1_macro"]),
        "balanced_accuracy": float(best["inner_val_balanced_accuracy"]),
        "recall_attack": float(best["inner_val_recall_attack"]),
        "recall_benign": float(best["inner_val_recall_benign"]),
        "roc_auc": best.get("inner_val_roc_auc"),
        "pr_auc": best.get("inner_val_pr_auc"),
    }

    save_json(fold_summary, fold_dir / "fold_summary.json")
    log.info(
        "[%s] threshold=%.3f | accuracy=%.4f f1_macro=%.4f recall_attack=%.4f",
        fold_name,
        best_threshold,
        fold_summary["accuracy"],
        fold_summary["f1_macro"],
        fold_summary["recall_attack"],
    )

    return {
        "fold_summary": fold_summary,
        "pred_df": pred_df,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LOSO XGBoost on IoT-23")
    parser.add_argument("--data_file", required=True, help="Path to all_flows.parquet")
    parser.add_argument("--out_dir", required=True, help="Directory for LOSO outputs")
    parser.add_argument("--target_col", default="label_binary", help="Binary target column")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--val_frac", type=float, default=0.20, help="Fraction of outer-train scenarios used for inner validation")
    parser.add_argument("--no_threshold_tuning", action="store_true", help="Disable inner validation threshold tuning")
    args = parser.parse_args()

    data_file = Path(args.data_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_all_flows(data_file)
    validate_all_flows(df, args.target_col)

    scenarios = sorted(df["scenario"].astype(str).unique().tolist())
    splits = generate_loso_splits(scenarios)

    log.info("Loaded %d rows from %s", len(df), data_file)
    log.info("Discovered %d scenarios for LOSO", len(scenarios))

    fold_rows = []
    pred_frames = []

    for split in splits:
        result = run_fold(
            df=df,
            split=split,
            out_dir=out_dir,
            target_col=args.target_col,
            seed=args.seed,
            val_frac=args.val_frac,
            tune_threshold=not args.no_threshold_tuning,
        )
        fold_rows.append(result["fold_summary"])
        pred_frames.append(result["pred_df"])

    folds_df = pd.DataFrame(fold_rows).sort_values("scenario").reset_index(drop=True)
    all_preds = pd.concat(pred_frames, ignore_index=True)

    folds_df.to_csv(out_dir / "loso_fold_metrics.csv", index=False)
    all_preds.to_parquet(out_dir / "loso_all_predictions.parquet", index=False)

    aggregate = compute_aggregate_summary(all_preds=all_preds, folds_df=folds_df)
    run_meta = {
        "data_file": str(data_file),
        "out_dir": str(out_dir),
        "target_col": args.target_col,
        "seed": int(args.seed),
        "val_frac": float(args.val_frac),
        "threshold_tuning": bool(not args.no_threshold_tuning),
        "n_rows": int(len(df)),
        "n_scenarios": int(len(scenarios)),
        "feature_cols": FEATURE_COLS,
        "numeric_cols": NUMERIC_COLS,
        "categorical_cols": CATEGORICAL_COLS,
        "param_grid": make_param_grid(),
    }

    save_json(aggregate, out_dir / "loso_summary.json")
    save_json(run_meta, out_dir / "run_meta.json")

    log.info("Saved fold metrics to %s", out_dir / "loso_fold_metrics.csv")
    log.info("Saved pooled predictions to %s", out_dir / "loso_all_predictions.parquet")
    log.info("Saved LOSO summary to %s", out_dir / "loso_summary.json")
    log.info(
        "LOSO mean F1 macro: %.4f ± %.4f",
        aggregate["fold_mean"]["f1_macro"],
        aggregate["fold_std"]["f1_macro"],
    )
    log.info(
        "LOSO mean recall_attack: %.4f ± %.4f",
        aggregate["fold_mean"]["recall_attack"],
        aggregate["fold_std"]["recall_attack"],
    )


if __name__ == "__main__":
    main()