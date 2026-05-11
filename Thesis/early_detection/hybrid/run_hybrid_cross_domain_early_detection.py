from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

try:
    from .hybrid_early_detection_common import (
        DEFAULT_FRACTIONS,
        IOT23_META_COLS,
        UNSW_META_COLS,
        add_temporal_context_features,
        build_aligned_frame,
        build_feature_mappings,
        evaluate_hybrid_iot23_target_split,
        evaluate_hybrid_unsw_target_split,
        infer_column_types,
        load_alignment_table,
        load_iot23_eval_frame,
        load_iot23_source_train,
        load_unsw_frame,
        maybe_sample_rows,
        prepare_unsw_eval_frame,
        save_json,
        split_unsw_train_val,
        train_interpretable_hybrid,
    )
except ImportError:
    import sys

    THIS_DIR = Path(__file__).resolve().parent
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))
    from hybrid_early_detection_common import (
        DEFAULT_FRACTIONS,
        IOT23_META_COLS,
        UNSW_META_COLS,
        add_temporal_context_features,
        build_aligned_frame,
        build_feature_mappings,
        evaluate_hybrid_iot23_target_split,
        evaluate_hybrid_unsw_target_split,
        infer_column_types,
        load_alignment_table,
        load_iot23_eval_frame,
        load_iot23_source_train,
        load_unsw_frame,
        maybe_sample_rows,
        prepare_unsw_eval_frame,
        save_json,
        split_unsw_train_val,
        train_interpretable_hybrid,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interpretable hybrid source-only cross-domain early detection using aligned "
            "features, prefix-aware temporal context, prototype similarity, and explicit gating."
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
        default="early_detection/hybrid/outputs_hybrid_exp1",
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
        default=100000,
        help="Optional cap on IoT-23 source-train rows.",
    )
    parser.add_argument(
        "--unsw_train_max_rows",
        type=int,
        default=100000,
        help="Optional cap on UNSW-NB15 source-train rows.",
    )
    parser.add_argument(
        "--iot_eval_max_rows_per_scenario",
        type=int,
        default=50000,
        help="Optional cap on earliest IoT-23 evaluation rows kept per scenario after sorting.",
    )
    parser.add_argument(
        "--unsw_eval_max_rows",
        type=int,
        default=30000,
        help="Optional cap on earliest UNSW-NB15 validation and test rows kept for evaluation.",
    )
    parser.add_argument(
        "--tabular_hidden_layers",
        nargs="+",
        type=int,
        default=[128, 64],
        help="Hidden layer sizes for the tabular branch.",
    )
    parser.add_argument(
        "--temporal_hidden_layers",
        nargs="+",
        type=int,
        default=[128, 64],
        help="Hidden layer sizes for the temporal branch.",
    )
    parser.add_argument("--mlp_alpha", type=float, default=0.0001, help="MLP L2 penalty.")
    parser.add_argument("--mlp_batch_size", type=int, default=512, help="MLP batch size.")
    parser.add_argument("--mlp_max_iter", type=int, default=40, help="MLP max iterations.")
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
    return summary_df.assign(
        direction=direction,
        source_dataset=source_dataset,
        target_dataset=target_dataset,
        source_train_rows=source_train_rows,
        target_val_rows_config=target_val_rows,
        target_test_rows_config=target_test_rows,
        n_aligned_features=n_aligned_features,
        model="hybrid_interpretable",
    )


def add_detail_metadata(detail_df: pd.DataFrame, direction: str, split_name: str) -> pd.DataFrame:
    detail_df = detail_df.copy()
    detail_df["direction"] = direction
    detail_df["split"] = split_name
    detail_df["model"] = "hybrid_interpretable"
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
    iot_mapping, unsw_mapping, aligned_feature_cols = build_feature_mappings(alignment_df)

    iot_train_cols = sorted(set(iot_mapping.keys()) | {"label"})
    iot_eval_cols = sorted(set(iot_mapping.keys()) | {"label"} | set(IOT23_META_COLS))
    unsw_base_cols = sorted(set(unsw_mapping.keys()) | {"label"} | set(UNSW_META_COLS))

    all_summaries: list[pd.DataFrame] = []
    all_iot_details: list[pd.DataFrame] = []
    all_unsw_details: list[pd.DataFrame] = []

    directions_to_run: list[str] = []
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
            source_train_df = build_aligned_frame(iot_train_raw, iot_mapping)

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
        else:
            source_dataset = "unsw"
            target_dataset = "iot23"

            unsw_source_train_raw = maybe_sample_rows(
                load_unsw_frame(unsw_train_csv, unsw_base_cols),
                args.unsw_train_max_rows,
                args.seed,
            )
            source_train_df = build_aligned_frame(unsw_source_train_raw, unsw_mapping)

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

        aligned_categorical_cols, aligned_numeric_cols = infer_column_types(source_train_df, aligned_feature_cols)
        source_train_df, temporal_numeric_cols = add_temporal_context_features(source_train_df, aligned_numeric_cols)
        target_val_df, _ = add_temporal_context_features(target_val_df, aligned_numeric_cols)
        target_test_df, _ = add_temporal_context_features(target_test_df, aligned_numeric_cols)
        temporal_feature_cols = aligned_feature_cols + temporal_numeric_cols

        predictor = train_interpretable_hybrid(
            source_df=source_train_df,
            aligned_feature_cols=aligned_feature_cols,
            aligned_categorical_cols=aligned_categorical_cols,
            aligned_numeric_cols=aligned_numeric_cols,
            temporal_feature_cols=temporal_feature_cols,
            seed=args.seed,
            tabular_hidden_layers=tuple(args.tabular_hidden_layers),
            temporal_hidden_layers=tuple(args.temporal_hidden_layers),
            alpha=args.mlp_alpha,
            batch_size=args.mlp_batch_size,
            max_iter=args.mlp_max_iter,
        )

        joblib.dump(predictor, direction_dir / "hybrid_predictor.joblib")
        pd.DataFrame(
            {
                "aligned_feature": aligned_feature_cols,
                "feature_type": [
                    "categorical" if col in aligned_categorical_cols else "numeric"
                    for col in aligned_feature_cols
                ],
            }
        ).to_csv(direction_dir / "used_features.csv", index=False)
        pd.DataFrame({"temporal_feature": temporal_feature_cols}).to_csv(
            direction_dir / "temporal_features.csv",
            index=False,
        )

        if target_dataset == "unsw":
            val_outputs = evaluate_hybrid_unsw_target_split("val", target_val_df, predictor, fractions, direction_dir)
            test_outputs = evaluate_hybrid_unsw_target_split("test", target_test_df, predictor, fractions, direction_dir)
            detail_key = "overall_unsw_attack_cat_summary.csv"
        else:
            val_outputs = evaluate_hybrid_iot23_target_split("val", target_val_df, predictor, fractions, direction_dir)
            test_outputs = evaluate_hybrid_iot23_target_split("test", target_test_df, predictor, fractions, direction_dir)
            detail_key = "overall_iot23_scenario_summary.csv"

        source_train_rows = int(len(source_train_df))
        target_val_rows = int(len(target_val_df))
        target_test_rows = int(len(target_test_df))

        direction_summary = pd.concat(
            [
                add_direction_metadata(
                    val_outputs["summary"],
                    direction_name,
                    source_dataset,
                    target_dataset,
                    source_train_rows,
                    target_val_rows,
                    target_test_rows,
                    len(aligned_feature_cols),
                ),
                add_direction_metadata(
                    test_outputs["summary"],
                    direction_name,
                    source_dataset,
                    target_dataset,
                    source_train_rows,
                    target_val_rows,
                    target_test_rows,
                    len(aligned_feature_cols),
                ),
            ],
            ignore_index=True,
        )
        direction_summary.to_csv(direction_dir / "overall_fraction_summary.csv", index=False)

        detail_df = pd.concat(
            [
                add_detail_metadata(val_outputs["details"], direction_name, "val"),
                add_detail_metadata(test_outputs["details"], direction_name, "test"),
            ],
            ignore_index=True,
        )
        detail_df.to_csv(direction_dir / detail_key, index=False)

        run_config = {
            "direction": direction_name,
            "source_dataset": source_dataset,
            "target_dataset": target_dataset,
            "fractions": fractions,
            "seed": args.seed,
            "source_train_rows": source_train_rows,
            "target_val_rows_config": target_val_rows,
            "target_test_rows_config": target_test_rows,
            "aligned_features": aligned_feature_cols,
            "aligned_categorical_features": aligned_categorical_cols,
            "aligned_numeric_features": aligned_numeric_cols,
            "temporal_feature_cols": temporal_feature_cols,
            "gating_style": "interpretable_rule_based",
            "tabular_hidden_layers": list(args.tabular_hidden_layers),
            "temporal_hidden_layers": list(args.temporal_hidden_layers),
            "mlp_alpha": args.mlp_alpha,
            "mlp_batch_size": args.mlp_batch_size,
            "mlp_max_iter": args.mlp_max_iter,
        }
        save_json(run_config, direction_dir / "run_config.json")

        all_summaries.append(direction_summary)
        if target_dataset == "unsw":
            all_unsw_details.append(detail_df)
        else:
            all_iot_details.append(detail_df)

    if all_summaries:
        pd.concat(all_summaries, ignore_index=True).to_csv(out_dir / "overall_fraction_summary.csv", index=False)
    if all_iot_details:
        pd.concat(all_iot_details, ignore_index=True).to_csv(out_dir / "overall_iot23_scenario_summary.csv", index=False)
    if all_unsw_details:
        pd.concat(all_unsw_details, ignore_index=True).to_csv(out_dir / "overall_unsw_attack_cat_summary.csv", index=False)


if __name__ == "__main__":
    main()
