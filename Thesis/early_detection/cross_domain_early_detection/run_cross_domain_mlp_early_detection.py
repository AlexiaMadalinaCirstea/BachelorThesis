from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from cross_domain_early_detection_common import (
    DEFAULT_FRACTIONS,
    IOT23_META_COLS,
    UNSW_META_COLS,
    build_aligned_frame,
    build_feature_mappings,
    build_mlp_pipeline,
    evaluate_iot23_target_split,
    evaluate_unsw_target_split,
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
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-domain early-detection MLP experiments between IoT-23 and UNSW-NB15 "
            "using the curated aligned feature subset."
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
        help="Path to the curated alignment CSV.",
    )
    parser.add_argument(
        "--out_dir",
        default="early_detection/cross_domain_early_detection/outputs_mlp_exp1",
        help="Directory for outputs.",
    )
    parser.add_argument(
        "--direction",
        choices=["iot23_to_unsw", "unsw_to_iot23", "both"],
        default="both",
        help="Which transfer direction to run.",
    )
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=DEFAULT_FRACTIONS,
        help="Prefix fractions to evaluate.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--include_review_features",
        action="store_true",
        help="Include rows marked review_required in the aligned feature set.",
    )
    parser.add_argument(
        "--unsw_val_fraction",
        type=float,
        default=0.2,
        help="Validation fraction carved from the UNSW training CSV when UNSW is the target.",
    )
    parser.add_argument(
        "--iot_train_max_rows",
        type=int,
        default=None,
        help="Optional cap on IoT-23 source-train rows.",
    )
    parser.add_argument(
        "--unsw_train_max_rows",
        type=int,
        default=None,
        help="Optional cap on UNSW-NB15 source-train rows.",
    )
    parser.add_argument(
        "--iot_eval_max_rows_per_scenario",
        type=int,
        default=None,
        help="Optional cap on earliest IoT-23 evaluation rows kept per scenario after sorting.",
    )
    parser.add_argument(
        "--unsw_eval_max_rows",
        type=int,
        default=None,
        help="Optional cap on earliest UNSW-NB15 validation and test rows kept for evaluation.",
    )
    parser.add_argument(
        "--mlp_hidden_layers",
        nargs="+",
        type=int,
        default=[128, 64],
        help="Hidden layer sizes for the MLP.",
    )
    parser.add_argument(
        "--mlp_alpha",
        type=float,
        default=0.0001,
        help="L2 penalty.",
    )
    parser.add_argument(
        "--mlp_max_iter",
        type=int,
        default=40,
        help="Maximum MLP iterations.",
    )
    parser.add_argument(
        "--mlp_batch_size",
        type=int,
        default=512,
        help="MLP batch size.",
    )
    return parser.parse_args()


def add_direction_metadata(
    summary_df: pd.DataFrame,
    direction: str,
    source_dataset: str,
    target_dataset: str,
    source_train_rows: int,
    target_val_rows: int,
    target_test_rows: int,
    n_aligned_features: int,
) -> pd.DataFrame:
    summary_df = summary_df.copy()
    summary_df["direction"] = direction
    summary_df["source_dataset"] = source_dataset
    summary_df["target_dataset"] = target_dataset
    summary_df["source_train_rows"] = source_train_rows
    summary_df["target_val_rows_config"] = target_val_rows
    summary_df["target_test_rows_config"] = target_test_rows
    summary_df["n_aligned_features"] = n_aligned_features
    return summary_df


def add_detail_metadata(detail_df: pd.DataFrame, direction: str, split_name: str, model: str) -> pd.DataFrame:
    detail_df = detail_df.copy()
    detail_df["direction"] = direction
    detail_df["split"] = split_name
    detail_df["model"] = model
    return detail_df


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

    all_summaries: list[pd.DataFrame] = []
    all_iot_details: list[pd.DataFrame] = []
    all_unsw_details: list[pd.DataFrame] = []
    run_counts: dict[str, int | None] = {
        "iot_train_rows": None,
        "iot_val_rows": None,
        "iot_test_rows": None,
        "unsw_source_train_rows": None,
        "unsw_target_val_rows": None,
        "unsw_test_rows": None,
    }

    directions_to_run = []
    if args.direction in {"iot23_to_unsw", "both"}:
        directions_to_run.append("iot23_to_unsw")
    if args.direction in {"unsw_to_iot23", "both"}:
        directions_to_run.append("unsw_to_iot23")

    for direction_name in directions_to_run:
        direction_dir = out_dir / direction_name
        direction_dir.mkdir(parents=True, exist_ok=True)

        if direction_name == "iot23_to_unsw":
            source_dataset = "iot23"
            target_dataset = "unsw"

            iot_train_raw = load_iot23_source_train(
                iot_data_dir / "train.parquet",
                iot_train_cols,
                args.iot_train_max_rows,
                args.seed,
            )
            unsw_full_raw = load_unsw_frame(unsw_train_csv, unsw_base_cols)
            _, unsw_val_raw = split_unsw_train_val(unsw_full_raw, args.unsw_val_fraction, args.seed)
            unsw_test_raw = load_unsw_frame(unsw_test_csv, unsw_base_cols)
            unsw_val_raw = prepare_unsw_eval_frame(unsw_val_raw, args.unsw_eval_max_rows)
            unsw_test_raw = prepare_unsw_eval_frame(unsw_test_raw, args.unsw_eval_max_rows)

            source_train_df = build_aligned_frame(iot_train_raw, iot_mapping)
            target_val_df = build_aligned_frame(unsw_val_raw, unsw_mapping, meta_cols=UNSW_META_COLS)
            target_test_df = build_aligned_frame(unsw_test_raw, unsw_mapping, meta_cols=UNSW_META_COLS)
            target_type = "unsw"

            run_counts["iot_train_rows"] = int(len(source_train_df))
            run_counts["unsw_target_val_rows"] = int(len(target_val_df))
            run_counts["unsw_test_rows"] = int(len(target_test_df))
        else:
            source_dataset = "unsw"
            target_dataset = "iot23"

            unsw_source_train_raw = maybe_sample_rows(
                load_unsw_frame(unsw_train_csv, unsw_base_cols),
                args.unsw_train_max_rows,
                args.seed,
            )
            iot_val_raw = load_iot23_eval_frame(
                iot_data_dir / "val.parquet",
                iot_eval_cols,
                args.iot_eval_max_rows_per_scenario,
            )
            iot_test_raw = load_iot23_eval_frame(
                iot_data_dir / "test.parquet",
                iot_eval_cols,
                args.iot_eval_max_rows_per_scenario,
            )

            source_train_df = build_aligned_frame(unsw_source_train_raw, unsw_mapping)
            target_val_df = build_aligned_frame(iot_val_raw, iot_mapping, meta_cols=IOT23_META_COLS)
            target_test_df = build_aligned_frame(iot_test_raw, iot_mapping, meta_cols=IOT23_META_COLS)
            target_type = "iot23"

            run_counts["unsw_source_train_rows"] = int(len(source_train_df))
            run_counts["iot_val_rows"] = int(len(target_val_df))
            run_counts["iot_test_rows"] = int(len(target_test_df))

        categorical_cols, numeric_cols = infer_column_types(source_train_df, feature_cols)
        pipeline = build_mlp_pipeline(
            categorical_cols=categorical_cols,
            numeric_cols=numeric_cols,
            hidden_layers=tuple(args.mlp_hidden_layers),
            alpha=args.mlp_alpha,
            max_iter=args.mlp_max_iter,
            batch_size=args.mlp_batch_size,
            seed=args.seed,
        )

        X_train = normalize_categorical_columns(source_train_df[feature_cols].copy(), categorical_cols)
        y_train = source_train_df["label"].copy()
        pipeline.fit(X_train, y_train)

        joblib.dump(pipeline, direction_dir / "mlp_pipeline.joblib")
        pd.DataFrame({"feature": feature_cols}).to_csv(direction_dir / "used_features.csv", index=False)

        if target_type == "unsw":
            val_results = evaluate_unsw_target_split("val", target_val_df, pipeline, feature_cols, categorical_cols, fractions, direction_dir)
            test_results = evaluate_unsw_target_split("test", target_test_df, pipeline, feature_cols, categorical_cols, fractions, direction_dir)
            all_unsw_details.append(add_detail_metadata(val_results["details"], direction_name, "val", "mlp"))
            all_unsw_details.append(add_detail_metadata(test_results["details"], direction_name, "test", "mlp"))
        else:
            val_results = evaluate_iot23_target_split("val", target_val_df, pipeline, feature_cols, categorical_cols, fractions, direction_dir)
            test_results = evaluate_iot23_target_split("test", target_test_df, pipeline, feature_cols, categorical_cols, fractions, direction_dir)
            all_iot_details.append(add_detail_metadata(val_results["details"], direction_name, "val", "mlp"))
            all_iot_details.append(add_detail_metadata(test_results["details"], direction_name, "test", "mlp"))

        all_summaries.append(
            add_direction_metadata(
                val_results["summary"],
                direction_name,
                source_dataset,
                target_dataset,
                len(source_train_df),
                len(target_val_df),
                len(target_test_df),
                len(feature_cols),
            ).assign(model="mlp")
        )
        all_summaries.append(
            add_direction_metadata(
                test_results["summary"],
                direction_name,
                source_dataset,
                target_dataset,
                len(source_train_df),
                len(target_val_df),
                len(target_test_df),
                len(feature_cols),
            ).assign(model="mlp")
        )

    pd.concat(all_summaries, ignore_index=True).to_csv(out_dir / "overall_fraction_summary.csv", index=False)

    if all_iot_details:
        pd.concat(all_iot_details, ignore_index=True).to_csv(out_dir / "overall_iot23_scenario_summary.csv", index=False)
    if all_unsw_details:
        pd.concat(all_unsw_details, ignore_index=True).to_csv(out_dir / "overall_unsw_attack_cat_summary.csv", index=False)

    save_json(
        {
            "model": "mlp",
            "direction": args.direction,
            "iot_data_dir": str(iot_data_dir),
            "unsw_train_csv": str(unsw_train_csv),
            "unsw_test_csv": str(unsw_test_csv),
            "alignment_csv": args.alignment_csv,
            "include_review_features": args.include_review_features,
            "fractions": fractions,
            "seed": args.seed,
            "unsw_val_fraction": args.unsw_val_fraction,
            "iot_train_max_rows": args.iot_train_max_rows,
            "unsw_train_max_rows": args.unsw_train_max_rows,
            "iot_eval_max_rows_per_scenario": args.iot_eval_max_rows_per_scenario,
            "unsw_eval_max_rows": args.unsw_eval_max_rows,
            "mlp_hidden_layers": list(args.mlp_hidden_layers),
            "mlp_alpha": args.mlp_alpha,
            "mlp_max_iter": args.mlp_max_iter,
            "mlp_batch_size": args.mlp_batch_size,
            "n_aligned_features": len(feature_cols),
            "aligned_features": feature_cols,
            "iot23_to_aligned_mapping": iot_mapping,
            "unsw_to_aligned_mapping": unsw_mapping,
            "iot_train_rows": run_counts["iot_train_rows"],
            "iot_val_rows": run_counts["iot_val_rows"],
            "iot_test_rows": run_counts["iot_test_rows"],
            "unsw_source_train_rows": run_counts["unsw_source_train_rows"],
            "unsw_target_val_rows": run_counts["unsw_target_val_rows"],
            "unsw_test_rows": run_counts["unsw_test_rows"],
        },
        out_dir / "run_config.json",
    )


if __name__ == "__main__":
    main()
