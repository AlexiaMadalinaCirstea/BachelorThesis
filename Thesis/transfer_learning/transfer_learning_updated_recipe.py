from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

import run_transfer_learning as base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Updated transfer-learning recipe for IoT-23 and UNSW-NB15. "
            "This version adds target-side calibration and threshold tuning to combat "
            "the collapse observed in the UNSW->IoT-23 direction."
        )
    )
    parser.add_argument("--iot_train", default="Datasets/IoT23/processed_full/iot23/train.parquet")
    parser.add_argument("--iot_test", default="Datasets/IoT23/processed_full/iot23/test.parquet")
    parser.add_argument(
        "--unsw_train",
        default="Datasets/UNSW-NB15/UNSW-NB15 dataset/CSV Files/Training and Testing Sets/UNSW_NB15_training-set.csv",
    )
    parser.add_argument(
        "--unsw_test",
        default="Datasets/UNSW-NB15/UNSW-NB15 dataset/CSV Files/Training and Testing Sets/UNSW_NB15_testing-set.csv",
    )
    parser.add_argument(
        "--alignment_csv",
        default="feature_alignment/comparison_outputs/aligned_features_curated.csv",
    )
    parser.add_argument("--out_dir", default="transfer_learning/outputs_updated_recipe")
    parser.add_argument("--include_review_features", action="store_true")
    parser.add_argument(
        "--target_fractions",
        nargs="+",
        type=float,
        default=[0.05, 0.10, 0.25, 0.50, 1.0],
    )
    parser.add_argument("--iot_train_max_rows", type=int, default=None)
    parser.add_argument("--iot_test_max_rows", type=int, default=None)
    parser.add_argument("--unsw_train_max_rows", type=int, default=None)
    parser.add_argument("--unsw_test_max_rows", type=int, default=None)
    parser.add_argument(
        "--balance_target_train",
        action="store_true",
        help="Apply balanced sampling before creating target fit/calibration splits.",
    )
    parser.add_argument(
        "--target_balance_ratio",
        type=float,
        default=1.0,
        help="Desired majority-to-minority ratio after balancing.",
    )
    parser.add_argument(
        "--calibration_fraction",
        type=float,
        default=0.20,
        help="Fraction of the target subset reserved for threshold selection.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
    )
    parser.add_argument(
        "--pretrain_estimators",
        type=int,
        default=50,
        help="Smaller source stage to reduce over-dominance of the source model.",
    )
    parser.add_argument(
        "--adapt_estimators",
        type=int,
        default=150,
        help="Larger target adaptation stage to let the target domain reshape the model.",
    )
    parser.add_argument(
        "--target_only_estimators",
        type=int,
        default=150,
        help="Trees for target-only training.",
    )
    parser.add_argument("--xgb_max_depth", type=int, default=6)
    parser.add_argument("--xgb_learning_rate", type=float, default=0.05)
    parser.add_argument("--random_state", type=int, default=42)
    return parser.parse_args()


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_attack": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_attack": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_attack": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }


def choose_best_threshold(
    y_true: pd.Series,
    y_score: pd.Series,
    thresholds: list[float],
) -> tuple[float, dict[str, float], pd.DataFrame]:
    rows = []
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        metrics = compute_metrics(y_true, y_pred)
        rows.append({"threshold": threshold, **metrics})

    threshold_df = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
    best_row = threshold_df.sort_values(
        ["f1_macro", "f1_attack", "accuracy"],
        ascending=[False, False, False],
    ).iloc[0]
    best_threshold = float(best_row["threshold"])
    best_metrics = {k: float(best_row[k]) for k in threshold_df.columns if k != "threshold"}
    return best_threshold, best_metrics, threshold_df


def default_threshold_summary(default_threshold: float = 0.5) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "threshold": default_threshold,
                "selection_mode": "default_no_calibration",
            }
        ]
    )


def split_target_train_for_calibration(
    df: pd.DataFrame,
    calibration_fraction: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    if calibration_fraction <= 0 or calibration_fraction >= 1:
        raise ValueError("calibration_fraction must be in (0, 1).")

    counts = df["label"].value_counts()
    if len(counts) < 2 or counts.min() < 2:
        return df.reset_index(drop=True), pd.DataFrame(columns=df.columns), False

    fit_df, calib_df = train_test_split(
        df,
        test_size=calibration_fraction,
        random_state=random_state,
        stratify=df["label"],
    )
    return fit_df.reset_index(drop=True), calib_df.reset_index(drop=True), True


def save_run_artifacts(
    out_dir: Path,
    y_true: pd.Series,
    y_pred: pd.Series,
    y_score: pd.Series,
    feature_names: list[str],
    model,
    metadata: dict[str, object],
    threshold_df: pd.DataFrame,
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

    threshold_df.to_csv(out_dir / "threshold_summary.csv", index=False)
    base.save_feature_importance(model, feature_names, out_dir / "feature_importance.csv")

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def prepare_encoded_views(
    fit_df: pd.DataFrame,
    calib_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[object, object, object, list[str], list[str], list[str]]:
    categorical_cols, numeric_cols = base.infer_column_types(fit_df, feature_cols)

    fit_x = base.normalize_categorical_columns(fit_df[feature_cols].copy(), categorical_cols)
    calib_x = base.normalize_categorical_columns(calib_df[feature_cols].copy(), categorical_cols)
    test_x = base.normalize_categorical_columns(test_df[feature_cols].copy(), categorical_cols)

    preprocessor = base.build_preprocessor(categorical_cols, numeric_cols)
    fit_encoded = preprocessor.fit_transform(fit_x)
    calib_encoded = preprocessor.transform(calib_x)
    test_encoded = preprocessor.transform(test_x)
    feature_names = base.transformed_feature_names(categorical_cols, numeric_cols)
    return fit_encoded, calib_encoded, test_encoded, feature_names, categorical_cols, numeric_cols


def evaluate_target_only_updated(
    target_fit_df: pd.DataFrame,
    target_calib_df: pd.DataFrame,
    target_test_df: pd.DataFrame,
    feature_cols: list[str],
    direction_label: str,
    fraction: float,
    direction_dir: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    fit_x, calib_x, test_x, feature_names, categorical_cols, numeric_cols = prepare_encoded_views(
        fit_df=target_fit_df,
        calib_df=target_calib_df,
        test_df=target_test_df,
        feature_cols=feature_cols,
    )

    model = base.build_xgb_model(n_estimators=args.target_only_estimators, args=args)
    y_fit = target_fit_df["label"].copy()
    y_calib = target_calib_df["label"].copy()
    y_test = target_test_df["label"].copy()

    model.fit(fit_x, y_fit)
    if len(target_calib_df) > 0:
        calib_score = pd.Series(model.predict_proba(calib_x)[:, 1])
        best_threshold, _, threshold_df = choose_best_threshold(y_calib, calib_score, args.thresholds)
        threshold_selection_mode = "calibration_split"
        n_target_calibration = int(len(target_calib_df))
    else:
        best_threshold = 0.5
        threshold_df = default_threshold_summary(default_threshold=best_threshold)
        threshold_selection_mode = "default_no_calibration"
        n_target_calibration = 0

    test_score = pd.Series(model.predict_proba(test_x)[:, 1])
    test_pred = (test_score >= best_threshold).astype(int)
    test_metrics = compute_metrics(y_test, test_pred)

    metadata = {
        "condition": "target_only_updated",
        "direction": direction_label,
        "target_fraction": fraction,
        "n_target_fit": int(len(target_fit_df)),
        "n_target_calibration": n_target_calibration,
        "n_target_test": int(len(target_test_df)),
        "n_features": int(len(feature_cols)),
        "categorical_features": categorical_cols,
        "numeric_features": numeric_cols,
        "target_only_estimators": args.target_only_estimators,
        "selected_threshold": best_threshold,
        "threshold_selection_mode": threshold_selection_mode,
        "metrics": test_metrics,
    }

    run_dir = direction_dir / f"target_only_updated_{base.fraction_to_slug(fraction)}"
    save_run_artifacts(
        out_dir=run_dir,
        y_true=y_test,
        y_pred=test_pred,
        y_score=test_score,
        feature_names=feature_names,
        model=model,
        metadata=metadata,
        threshold_df=threshold_df,
    )

    return {
        "direction": direction_label,
        "condition": "target_only_updated",
        "target_fraction": fraction,
        "n_source_train": 0,
        "n_target_fit": int(len(target_fit_df)),
        "n_target_calibration": n_target_calibration,
        "n_target_test": int(len(target_test_df)),
        "selected_threshold": best_threshold,
        "threshold_selection_mode": threshold_selection_mode,
        **test_metrics,
    }


def evaluate_transfer_learning_updated(
    source_train_df: pd.DataFrame,
    target_fit_df: pd.DataFrame,
    target_calib_df: pd.DataFrame,
    target_test_df: pd.DataFrame,
    feature_cols: list[str],
    direction_label: str,
    fraction: float,
    direction_dir: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    categorical_cols, numeric_cols = base.infer_column_types(source_train_df, feature_cols)

    source_x_df = base.normalize_categorical_columns(source_train_df[feature_cols].copy(), categorical_cols)
    target_fit_x_df = base.normalize_categorical_columns(target_fit_df[feature_cols].copy(), categorical_cols)
    target_calib_x_df = base.normalize_categorical_columns(target_calib_df[feature_cols].copy(), categorical_cols)
    target_test_x_df = base.normalize_categorical_columns(target_test_df[feature_cols].copy(), categorical_cols)

    preprocessor = base.build_preprocessor(categorical_cols, numeric_cols)
    source_x = preprocessor.fit_transform(source_x_df)
    target_fit_x = preprocessor.transform(target_fit_x_df)
    target_calib_x = preprocessor.transform(target_calib_x_df)
    target_test_x = preprocessor.transform(target_test_x_df)
    feature_names = base.transformed_feature_names(categorical_cols, numeric_cols)

    y_source = source_train_df["label"].copy()
    y_target_fit = target_fit_df["label"].copy()
    y_target_calib = target_calib_df["label"].copy()
    y_target_test = target_test_df["label"].copy()

    source_model = base.build_xgb_model(n_estimators=args.pretrain_estimators, args=args)
    source_model.fit(source_x, y_source)

    adapted_model = base.build_xgb_model(n_estimators=args.adapt_estimators, args=args)
    adapted_model.fit(target_fit_x, y_target_fit, xgb_model=source_model.get_booster())

    if len(target_calib_df) > 0:
        calib_score = pd.Series(adapted_model.predict_proba(target_calib_x)[:, 1])
        best_threshold, _, threshold_df = choose_best_threshold(y_target_calib, calib_score, args.thresholds)
        threshold_selection_mode = "calibration_split"
        n_target_calibration = int(len(target_calib_df))
    else:
        best_threshold = 0.5
        threshold_df = default_threshold_summary(default_threshold=best_threshold)
        threshold_selection_mode = "default_no_calibration"
        n_target_calibration = 0

    test_score = pd.Series(adapted_model.predict_proba(target_test_x)[:, 1])
    test_pred = (test_score >= best_threshold).astype(int)
    test_metrics = compute_metrics(y_target_test, test_pred)

    metadata = {
        "condition": "transfer_learning_updated",
        "direction": direction_label,
        "target_fraction": fraction,
        "n_source_train": int(len(source_train_df)),
        "n_target_fit": int(len(target_fit_df)),
        "n_target_calibration": n_target_calibration,
        "n_target_test": int(len(target_test_df)),
        "n_features": int(len(feature_cols)),
        "categorical_features": categorical_cols,
        "numeric_features": numeric_cols,
        "pretrain_estimators": args.pretrain_estimators,
        "adapt_estimators": args.adapt_estimators,
        "selected_threshold": best_threshold,
        "threshold_selection_mode": threshold_selection_mode,
        "metrics": test_metrics,
    }

    run_dir = direction_dir / f"transfer_learning_updated_{base.fraction_to_slug(fraction)}"
    save_run_artifacts(
        out_dir=run_dir,
        y_true=y_target_test,
        y_pred=test_pred,
        y_score=test_score,
        feature_names=feature_names,
        model=adapted_model,
        metadata=metadata,
        threshold_df=threshold_df,
    )
    base.save_feature_importance(source_model, feature_names, run_dir / "feature_importance_pretrain.csv")

    return {
        "direction": direction_label,
        "condition": "transfer_learning_updated",
        "target_fraction": fraction,
        "n_source_train": int(len(source_train_df)),
        "n_target_fit": int(len(target_fit_df)),
        "n_target_calibration": n_target_calibration,
        "n_target_test": int(len(target_test_df)),
        "selected_threshold": best_threshold,
        "threshold_selection_mode": threshold_selection_mode,
        **test_metrics,
    }


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    alignment_df = base.load_alignment_table(
        Path(args.alignment_csv),
        include_review_features=args.include_review_features,
    )

    iot_required_cols = sorted(set(alignment_df["iot23_feature"].tolist() + ["label"]))
    unsw_required_cols = sorted(set(alignment_df["unsw_feature"].tolist() + ["label"]))

    iot_train = base.load_iot23(Path(args.iot_train), columns=iot_required_cols, max_rows=args.iot_train_max_rows, random_state=args.random_state)
    iot_test = base.load_iot23(Path(args.iot_test), columns=iot_required_cols, max_rows=args.iot_test_max_rows, random_state=args.random_state)
    unsw_train = base.load_unsw(Path(args.unsw_train), columns=unsw_required_cols, max_rows=args.unsw_train_max_rows, random_state=args.random_state)
    unsw_test = base.load_unsw(Path(args.unsw_test), columns=unsw_required_cols, max_rows=args.unsw_test_max_rows, random_state=args.random_state)

    iot_train_aligned, unsw_train_aligned, feature_cols = base.build_aligned_views(
        iot_df=iot_train,
        unsw_df=unsw_train,
        alignment_df=alignment_df,
    )
    iot_test_aligned, unsw_test_aligned, _ = base.build_aligned_views(
        iot_df=iot_test,
        unsw_df=unsw_test,
        alignment_df=alignment_df,
    )

    with open(out_dir / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
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
                "balance_target_train": args.balance_target_train,
                "target_balance_ratio": args.target_balance_ratio,
                "calibration_fraction": args.calibration_fraction,
                "thresholds": args.thresholds,
                "pretrain_estimators": args.pretrain_estimators,
                "adapt_estimators": args.adapt_estimators,
                "target_only_estimators": args.target_only_estimators,
                "xgb_max_depth": args.xgb_max_depth,
                "xgb_learning_rate": args.xgb_learning_rate,
                "random_state": args.random_state,
                "n_aligned_features": len(feature_cols),
                "aligned_features": feature_cols,
            },
            handle,
            indent=2,
        )

    summary_rows = []
    directions = [
        ("iot23_train", "unsw", iot_train_aligned, unsw_train_aligned, unsw_test_aligned),
        ("unsw_train", "iot23", unsw_train_aligned, iot_train_aligned, iot_test_aligned),
    ]

    for source_name, target_name, source_train_df, target_train_df, target_test_df in directions:
        direction_label = f"{source_name}->{target_name}_test"
        direction_dir = out_dir / base.direction_to_slug(source_name=source_name, target_name=target_name)
        direction_dir.mkdir(parents=True, exist_ok=True)

        for fraction in args.target_fractions:
            sampled_target_train = base.sample_target_fraction(
                df=target_train_df,
                fraction=fraction,
                random_state=args.random_state,
            )
            if args.balance_target_train:
                sampled_target_train = base.balance_binary_target_train(
                    df=sampled_target_train,
                    random_state=args.random_state,
                    majority_to_minority_ratio=args.target_balance_ratio,
                )

            target_fit_df, target_calib_df, _ = split_target_train_for_calibration(
                df=sampled_target_train,
                calibration_fraction=args.calibration_fraction,
                random_state=args.random_state,
            )

            summary_rows.append(
                evaluate_target_only_updated(
                    target_fit_df=target_fit_df,
                    target_calib_df=target_calib_df,
                    target_test_df=target_test_df,
                    feature_cols=feature_cols,
                    direction_label=direction_label,
                    fraction=fraction,
                    direction_dir=direction_dir,
                    args=args,
                )
            )
            summary_rows.append(
                evaluate_transfer_learning_updated(
                    source_train_df=source_train_df,
                    target_fit_df=target_fit_df,
                    target_calib_df=target_calib_df,
                    target_test_df=target_test_df,
                    feature_cols=feature_cols,
                    direction_label=direction_label,
                    fraction=fraction,
                    direction_dir=direction_dir,
                    args=args,
                )
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "updated_recipe_summary.csv", index=False)

    print("Updated transfer-learning recipe complete.")
    print(summary_df.to_string(index=False))
    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
