from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

try:
    from .transfer_learning_early_detection_common import (
        DEFAULT_FRACTIONS,
        IOT23_META_COLS,
        TRANSFER_CONDITIONS,
        UNSW_META_COLS,
        PreprocessedMLPPredictor,
        build_aligned_frame,
        build_feature_mappings,
        evaluate_iot23_condition_split,
        evaluate_unsw_condition_split,
        fit_preprocessor_from_frames,
        infer_column_types,
        load_alignment_table,
        load_iot23_eval_frame,
        load_iot23_source_train,
        load_unsw_frame,
        maybe_sample_rows,
        prepare_target_train_subset,
        prepare_unsw_eval_frame,
        save_json,
        split_unsw_train_val,
        train_mlp_incremental,
        transform_features,
    )
except ImportError:
    import sys

    THIS_DIR = Path(__file__).resolve().parent
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))
    from transfer_learning_early_detection_common import (
        DEFAULT_FRACTIONS,
        IOT23_META_COLS,
        TRANSFER_CONDITIONS,
        UNSW_META_COLS,
        PreprocessedMLPPredictor,
        build_aligned_frame,
        build_feature_mappings,
        evaluate_iot23_condition_split,
        evaluate_unsw_condition_split,
        fit_preprocessor_from_frames,
        infer_column_types,
        load_alignment_table,
        load_iot23_eval_frame,
        load_iot23_source_train,
        load_unsw_frame,
        maybe_sample_rows,
        prepare_target_train_subset,
        prepare_unsw_eval_frame,
        save_json,
        split_unsw_train_val,
        train_mlp_incremental,
        transform_features,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Transfer-learning early-detection experiment with matched source-only, "
            "target-only, and transfer-adapted MLP conditions."
        )
    )
    parser.add_argument(
        "--iot_data_dir",
        default="Datasets/IoT23/processed_full/iot23",
        help="Directory containing IoT-23 train/val/test parquet files.",
    )
    parser.add_argument(
        "--unsw_train_csv",
        default=r"Datasets/UNSW-NB15/UNSW-NB15 dataset/CSV Files/Training and Testing Sets/UNSW_NB15_training-set.csv",
        help="Path to the UNSW-NB15 training CSV.",
    )
    parser.add_argument(
        "--unsw_test_csv",
        default=r"Datasets/UNSW-NB15/UNSW-NB15 dataset/CSV Files/Training and Testing Sets/UNSW_NB15_testing-set.csv",
        help="Path to the UNSW-NB15 testing CSV.",
    )
    parser.add_argument(
        "--alignment_csv",
        default="feature_alignment/comparison_outputs/aligned_features_curated.csv",
        help="Path to the curated aligned-feature CSV.",
    )
    parser.add_argument(
        "--out_dir",
        default="early_detection/transfer_learning_based_early_detection/outputs_iot23_to_unsw_budget5k_seed42",
        help="Run output directory.",
    )
    parser.add_argument(
        "--direction",
        choices=["iot23_to_unsw", "unsw_to_iot23"],
        required=True,
        help="Transfer direction.",
    )
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=DEFAULT_FRACTIONS,
        help="Target evaluation prefix fractions.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--target_train_rows",
        type=int,
        required=True,
        help="Number of labeled target-train rows used for target-only and transfer-adapted conditions.",
    )
    parser.add_argument(
        "--source_train_rows",
        type=int,
        default=100000,
        help="Source-train row cap for the pretraining stage.",
    )
    parser.add_argument(
        "--include_review_features",
        action="store_true",
        help="Include review_required aligned features.",
    )
    parser.add_argument(
        "--unsw_val_fraction",
        type=float,
        default=0.2,
        help="UNSW validation fraction when UNSW is the target.",
    )
    parser.add_argument(
        "--iot_eval_max_rows_per_scenario",
        type=int,
        default=50000,
        help="Optional IoT target eval cap per scenario.",
    )
    parser.add_argument(
        "--unsw_eval_max_rows",
        type=int,
        default=30000,
        help="Optional UNSW target eval cap.",
    )
    parser.add_argument(
        "--mlp_hidden_layers",
        nargs="+",
        type=int,
        default=[128, 64],
        help="MLP hidden layer sizes.",
    )
    parser.add_argument("--mlp_alpha", type=float, default=0.0001, help="MLP L2 penalty.")
    parser.add_argument("--mlp_batch_size", type=int, default=512, help="MLP batch size.")
    parser.add_argument(
        "--source_epochs",
        type=int,
        default=20,
        help="Incremental epochs for source-only pretraining.",
    )
    parser.add_argument(
        "--target_only_epochs",
        type=int,
        default=20,
        help="Incremental epochs for target-only training.",
    )
    parser.add_argument(
        "--finetune_epochs",
        type=int,
        default=10,
        help="Additional incremental epochs for transfer adaptation on the target subset.",
    )
    return parser.parse_args()


def add_summary_metadata(
    summary_df: pd.DataFrame,
    condition: str,
    direction: str,
    source_dataset: str,
    target_dataset: str,
    source_train_rows: int,
    target_train_rows: int,
    target_val_rows: int,
    target_test_rows: int,
    n_aligned_features: int,
) -> pd.DataFrame:
    return summary_df.assign(
        condition=condition,
        direction=direction,
        source_dataset=source_dataset,
        target_dataset=target_dataset,
        source_train_rows=source_train_rows,
        target_train_rows=target_train_rows,
        target_val_rows_config=target_val_rows,
        target_test_rows_config=target_test_rows,
        n_aligned_features=n_aligned_features,
    )


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    iot_data_dir = Path(args.iot_data_dir)
    unsw_train_csv = Path(args.unsw_train_csv)
    unsw_test_csv = Path(args.unsw_test_csv)
    fractions = sorted(set(args.fractions))

    alignment_df = load_alignment_table(
        Path(args.alignment_csv),
        include_review_features=args.include_review_features,
    )
    iot_mapping, unsw_mapping, feature_cols = build_feature_mappings(alignment_df)

    iot_train_cols = sorted(set(iot_mapping.keys()) | {"label"})
    iot_eval_cols = sorted(set(iot_mapping.keys()) | {"label"} | set(IOT23_META_COLS))
    unsw_base_cols = sorted(set(unsw_mapping.keys()) | {"label"} | set(UNSW_META_COLS))

    if args.direction == "iot23_to_unsw":
        source_dataset = "iot23"
        target_dataset = "unsw"

        source_train_raw = load_iot23_source_train(
            iot_data_dir / "train.parquet",
            iot_train_cols,
            args.source_train_rows,
            args.seed,
        )
        source_train_df = build_aligned_frame(source_train_raw, iot_mapping)

        target_train_df = prepare_target_train_subset(
            direction=args.direction,
            iot_data_dir=iot_data_dir,
            unsw_train_csv=unsw_train_csv,
            budget_rows=args.target_train_rows,
            seed=args.seed,
            iot_mapping=iot_mapping,
            unsw_mapping=unsw_mapping,
        )

        unsw_full_raw = load_unsw_frame(unsw_train_csv, unsw_base_cols)
        _, unsw_val_raw = split_unsw_train_val(unsw_full_raw, args.unsw_val_fraction, args.seed)
        unsw_test_raw = load_unsw_frame(unsw_test_csv, unsw_base_cols)
        target_val_df = build_aligned_frame(
            prepare_unsw_eval_frame(unsw_val_raw, args.unsw_eval_max_rows),
            unsw_mapping,
            meta_cols=UNSW_META_COLS,
        )
        target_test_df = build_aligned_frame(
            prepare_unsw_eval_frame(unsw_test_raw, args.unsw_eval_max_rows),
            unsw_mapping,
            meta_cols=UNSW_META_COLS,
        )
        target_eval_kind = "unsw"
    else:
        source_dataset = "unsw"
        target_dataset = "iot23"

        unsw_source_raw = maybe_sample_rows(
            load_unsw_frame(unsw_train_csv, unsw_base_cols),
            args.source_train_rows,
            args.seed,
        )
        source_train_df = build_aligned_frame(unsw_source_raw, unsw_mapping)

        target_train_df = prepare_target_train_subset(
            direction=args.direction,
            iot_data_dir=iot_data_dir,
            unsw_train_csv=unsw_train_csv,
            budget_rows=args.target_train_rows,
            seed=args.seed,
            iot_mapping=iot_mapping,
            unsw_mapping=unsw_mapping,
        )

        target_val_df = build_aligned_frame(
            load_iot23_eval_frame(
                iot_data_dir / "val.parquet",
                iot_eval_cols,
                args.iot_eval_max_rows_per_scenario,
            ),
            iot_mapping,
            meta_cols=IOT23_META_COLS,
        )
        target_test_df = build_aligned_frame(
            load_iot23_eval_frame(
                iot_data_dir / "test.parquet",
                iot_eval_cols,
                args.iot_eval_max_rows_per_scenario,
            ),
            iot_mapping,
            meta_cols=IOT23_META_COLS,
        )
        target_eval_kind = "iot23"

    categorical_cols, numeric_cols = infer_column_types(source_train_df, feature_cols)
    source_preprocessor = fit_preprocessor_from_frames(
        frames=[source_train_df],
        feature_cols=feature_cols,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
    )
    target_preprocessor = fit_preprocessor_from_frames(
        frames=[target_train_df],
        feature_cols=feature_cols,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
    )

    X_source = transform_features(source_train_df, source_preprocessor, feature_cols, categorical_cols)
    y_source = source_train_df["label"]
    X_target_train_transfer = transform_features(target_train_df, source_preprocessor, feature_cols, categorical_cols)
    X_target_train_target_only = transform_features(target_train_df, target_preprocessor, feature_cols, categorical_cols)
    y_target_train = target_train_df["label"]

    source_model = train_mlp_incremental(
        X_source,
        y_source,
        hidden_layers=tuple(args.mlp_hidden_layers),
        alpha=args.mlp_alpha,
        batch_size=args.mlp_batch_size,
        seed=args.seed,
        epochs=args.source_epochs,
    )
    target_only_model = train_mlp_incremental(
        X_target_train_target_only,
        y_target_train,
        hidden_layers=tuple(args.mlp_hidden_layers),
        alpha=args.mlp_alpha,
        batch_size=args.mlp_batch_size,
        seed=args.seed,
        epochs=args.target_only_epochs,
    )
    transfer_model = train_mlp_incremental(
        X_target_train_transfer,
        y_target_train,
        hidden_layers=tuple(args.mlp_hidden_layers),
        alpha=args.mlp_alpha,
        batch_size=args.mlp_batch_size,
        seed=args.seed,
        epochs=args.finetune_epochs,
        initial_model=source_model,
    )

    condition_artifacts = {
        "source_only": {"preprocessor": source_preprocessor, "model": source_model},
        "target_only": {"preprocessor": target_preprocessor, "model": target_only_model},
        "transfer_adapted": {"preprocessor": source_preprocessor, "model": transfer_model},
    }

    joblib.dump(
        {
            "feature_cols": feature_cols,
            "categorical_cols": categorical_cols,
            "condition_artifacts": condition_artifacts,
        },
        out_dir / "condition_models.joblib",
    )

    all_summaries: list[pd.DataFrame] = []
    all_iot_details: list[pd.DataFrame] = []
    all_unsw_details: list[pd.DataFrame] = []

    for condition in TRANSFER_CONDITIONS:
        predictor = PreprocessedMLPPredictor(
            preprocessor=condition_artifacts[condition]["preprocessor"],
            model=condition_artifacts[condition]["model"],
            feature_cols=feature_cols,
            categorical_cols=categorical_cols,
        )

        if target_eval_kind == "iot23":
            val_results = evaluate_iot23_condition_split("val", target_val_df, predictor, fractions, out_dir, condition)
            test_results = evaluate_iot23_condition_split("test", target_test_df, predictor, fractions, out_dir, condition)
            all_iot_details.extend([val_results["details"], test_results["details"]])
        else:
            val_results = evaluate_unsw_condition_split("val", target_val_df, predictor, fractions, out_dir, condition)
            test_results = evaluate_unsw_condition_split("test", target_test_df, predictor, fractions, out_dir, condition)
            all_unsw_details.extend([val_results["details"], test_results["details"]])

        all_summaries.append(
            add_summary_metadata(
                val_results["summary"],
                condition=condition,
                direction=args.direction,
                source_dataset=source_dataset,
                target_dataset=target_dataset,
                source_train_rows=len(source_train_df),
                target_train_rows=len(target_train_df),
                target_val_rows=len(target_val_df),
                target_test_rows=len(target_test_df),
                n_aligned_features=len(feature_cols),
            )
        )
        all_summaries.append(
            add_summary_metadata(
                test_results["summary"],
                condition=condition,
                direction=args.direction,
                source_dataset=source_dataset,
                target_dataset=target_dataset,
                source_train_rows=len(source_train_df),
                target_train_rows=len(target_train_df),
                target_val_rows=len(target_val_df),
                target_test_rows=len(target_test_df),
                n_aligned_features=len(feature_cols),
            )
        )

    pd.concat(all_summaries, ignore_index=True).to_csv(out_dir / "overall_fraction_summary.csv", index=False)

    if all_iot_details:
        pd.concat(all_iot_details, ignore_index=True).to_csv(out_dir / "overall_iot23_scenario_summary.csv", index=False)
    if all_unsw_details:
        pd.concat(all_unsw_details, ignore_index=True).to_csv(out_dir / "overall_unsw_attack_cat_summary.csv", index=False)

    save_json(
        {
            "experiment_type": "transfer_learning_early_detection",
            "direction": args.direction,
            "conditions": TRANSFER_CONDITIONS,
            "seed": args.seed,
            "fractions": fractions,
            "target_train_rows": args.target_train_rows,
            "source_train_rows": len(source_train_df),
            "target_val_rows": len(target_val_df),
            "target_test_rows": len(target_test_df),
            "source_dataset": source_dataset,
            "target_dataset": target_dataset,
            "iot_data_dir": str(iot_data_dir),
            "unsw_train_csv": str(unsw_train_csv),
            "unsw_test_csv": str(unsw_test_csv),
            "alignment_csv": args.alignment_csv,
            "include_review_features": args.include_review_features,
            "unsw_val_fraction": args.unsw_val_fraction,
            "iot_eval_max_rows_per_scenario": args.iot_eval_max_rows_per_scenario,
            "unsw_eval_max_rows": args.unsw_eval_max_rows,
            "mlp_hidden_layers": list(args.mlp_hidden_layers),
            "mlp_alpha": args.mlp_alpha,
            "mlp_batch_size": args.mlp_batch_size,
            "source_epochs": args.source_epochs,
            "target_only_epochs": args.target_only_epochs,
            "finetune_epochs": args.finetune_epochs,
            "n_aligned_features": len(feature_cols),
            "aligned_features": feature_cols,
            "iot23_to_aligned_mapping": iot_mapping,
            "unsw_to_aligned_mapping": unsw_mapping,
        },
        out_dir / "run_config.json",
    )


if __name__ == "__main__":
    main()
