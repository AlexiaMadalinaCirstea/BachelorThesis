from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

try:
    from .hybrid_torch_common import (
        DEFAULT_FRACTIONS,
        DEVICE,
        EVIDENCE_PROGRESS_COL,
        PREFIX_ROWS_SEEN_COL,
        IOT23_META_COLS,
        UNSW_META_COLS,
        add_prefix_metadata,
        assert_valid_hybrid_frame,
        build_aligned_frame,
        build_feature_mappings,
        evaluate_full_hybrid_iot23_target_split,
        evaluate_full_hybrid_unsw_target_split,
        infer_column_types,
        load_alignment_table,
        load_iot23_eval_frame,
        load_iot23_source_train,
        load_unsw_frame,
        maybe_sample_rows,
        prepare_unsw_eval_frame,
        save_json,
        set_global_seeds,
        split_unsw_train_val,
        train_full_hybrid,
    )
except ImportError:
    import sys

    THIS_DIR = Path(__file__).resolve().parent
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))
    from hybrid_torch_common import (
        DEFAULT_FRACTIONS,
        DEVICE,
        EVIDENCE_PROGRESS_COL,
        PREFIX_ROWS_SEEN_COL,
        IOT23_META_COLS,
        UNSW_META_COLS,
        add_prefix_metadata,
        assert_valid_hybrid_frame,
        build_aligned_frame,
        build_feature_mappings,
        evaluate_full_hybrid_iot23_target_split,
        evaluate_full_hybrid_unsw_target_split,
        infer_column_types,
        load_alignment_table,
        load_iot23_eval_frame,
        load_iot23_source_train,
        load_unsw_frame,
        maybe_sample_rows,
        prepare_unsw_eval_frame,
        save_json,
        set_global_seeds,
        split_unsw_train_val,
        train_full_hybrid,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full Torch hybrid source-only cross-domain early detection."
    )
    parser.add_argument("--iot_data_dir", default="Datasets/IoT23/processed_full/iot23")
    parser.add_argument(
        "--unsw_train_csv",
        default=r"Datasets/UNSW-NB15/UNSW-NB15 dataset/CSV Files/Training and Testing Sets/UNSW_NB15_training-set.csv",
    )
    parser.add_argument(
        "--unsw_test_csv",
        default=r"Datasets/UNSW-NB15/UNSW-NB15 dataset/CSV Files/Training and Testing Sets/UNSW_NB15_testing-set.csv",
    )
    parser.add_argument(
        "--alignment_csv",
        default="feature_alignment/comparison_outputs/aligned_features_curated.csv",
    )
    parser.add_argument("--out_dir", default="early_detection/hybrid/outputs_hybrid_full_exp1")
    parser.add_argument("--direction", choices=["iot23_to_unsw", "unsw_to_iot23", "both"], default="both")
    parser.add_argument("--fractions", nargs="+", type=float, default=DEFAULT_FRACTIONS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--include_review_features", action="store_true")
    parser.add_argument("--unsw_val_fraction", type=float, default=0.2)
    parser.add_argument("--iot_train_max_rows", type=int, default=100000)
    parser.add_argument("--unsw_train_max_rows", type=int, default=100000)
    parser.add_argument("--iot_eval_max_rows_per_scenario", type=int, default=50000)
    parser.add_argument("--unsw_eval_max_rows", type=int, default=30000)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--train_val_fraction", type=float, default=0.15)
    parser.add_argument("--token_dim", type=int, default=64)
    parser.add_argument("--transformer_depth", type=int, default=2)
    parser.add_argument("--transformer_heads", type=int, default=4)
    parser.add_argument("--tcn_hidden_dim", type=int, default=64)
    parser.add_argument("--tcn_blocks", type=int, default=3)
    parser.add_argument("--prototype_embed_dim", type=int, default=64)
    parser.add_argument("--gate_hidden_dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--attention_dropout", type=float, default=None)
    parser.add_argument("--ffn_dropout", type=float, default=None)
    parser.add_argument("--residual_dropout", type=float, default=None)
    parser.add_argument("--disable_tabular_branch", action="store_true")
    parser.add_argument("--disable_temporal_branch", action="store_true")
    parser.add_argument("--disable_prototype_branch", action="store_true")
    parser.add_argument("--uniform_gating", action="store_true")
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
    ablation_name: str,
) -> pd.DataFrame:
    return summary_df.assign(
        direction=direction,
        source_dataset=source_dataset,
        target_dataset=target_dataset,
        source_train_rows=source_train_rows,
        target_val_rows_config=target_val_rows,
        target_test_rows_config=target_test_rows,
        n_aligned_features=n_aligned_features,
        model="hybrid_full_torch",
        ablation_name=ablation_name,
    )


def add_detail_metadata(detail_df: pd.DataFrame, direction: str, split_name: str, ablation_name: str) -> pd.DataFrame:
    detail_df = detail_df.copy()
    detail_df["direction"] = direction
    detail_df["split"] = split_name
    detail_df["model"] = "hybrid_full_torch"
    detail_df["ablation_name"] = ablation_name
    return detail_df


def resolve_ablation_name(args: argparse.Namespace) -> str:
    parts = []
    if args.disable_tabular_branch:
        parts.append("no_tabular")
    if args.disable_temporal_branch:
        parts.append("no_temporal")
    if args.disable_prototype_branch:
        parts.append("no_prototype")
    if args.uniform_gating:
        parts.append("uniform_gating")
    return "full_model" if not parts else "__".join(parts)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_global_seeds(args.seed, args.deterministic)

    iot_data_dir = Path(args.iot_data_dir)
    unsw_train_csv = Path(args.unsw_train_csv)
    unsw_test_csv = Path(args.unsw_test_csv)
    fractions = sorted(set(args.fractions))
    ablation_name = resolve_ablation_name(args)
    attention_dropout = args.dropout if args.attention_dropout is None else args.attention_dropout
    ffn_dropout = args.dropout if args.ffn_dropout is None else args.ffn_dropout
    residual_dropout = args.dropout if args.residual_dropout is None else args.residual_dropout

    alignment_df = load_alignment_table(Path(args.alignment_csv), args.include_review_features)
    iot_mapping, unsw_mapping, aligned_feature_cols = build_feature_mappings(alignment_df)

    iot_train_cols = sorted(set(iot_mapping.keys()) | {"label"})
    iot_eval_cols = sorted(set(iot_mapping.keys()) | {"label"} | set(IOT23_META_COLS))
    unsw_base_cols = sorted(set(unsw_mapping.keys()) | {"label"} | set(UNSW_META_COLS))

    all_summaries: list[pd.DataFrame] = []
    all_iot_details: list[pd.DataFrame] = []
    all_unsw_details: list[pd.DataFrame] = []

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
            source_train_df = build_aligned_frame(
                load_iot23_source_train(iot_data_dir / "train.parquet", iot_train_cols, args.iot_train_max_rows, args.seed),
                iot_mapping,
            )
            unsw_full = load_unsw_frame(unsw_train_csv, unsw_base_cols)
            _, unsw_val = split_unsw_train_val(unsw_full, args.unsw_val_fraction, args.seed)
            unsw_test = load_unsw_frame(unsw_test_csv, unsw_base_cols)
            target_val_df = build_aligned_frame(
                prepare_unsw_eval_frame(unsw_val, args.unsw_eval_max_rows),
                unsw_mapping,
                meta_cols=UNSW_META_COLS,
            )
            target_test_df = build_aligned_frame(
                prepare_unsw_eval_frame(unsw_test, args.unsw_eval_max_rows),
                unsw_mapping,
                meta_cols=UNSW_META_COLS,
            )
        else:
            source_dataset = "unsw"
            target_dataset = "iot23"
            source_train_df = build_aligned_frame(
                maybe_sample_rows(load_unsw_frame(unsw_train_csv, unsw_base_cols), args.unsw_train_max_rows, args.seed),
                unsw_mapping,
            )
            target_val_df = build_aligned_frame(
                load_iot23_eval_frame(iot_data_dir / "val.parquet", iot_eval_cols, args.iot_eval_max_rows_per_scenario),
                iot_mapping,
                meta_cols=IOT23_META_COLS,
            )
            target_test_df = build_aligned_frame(
                load_iot23_eval_frame(iot_data_dir / "test.parquet", iot_eval_cols, args.iot_eval_max_rows_per_scenario),
                iot_mapping,
                meta_cols=IOT23_META_COLS,
            )

        source_train_df = add_prefix_metadata(source_train_df)
        target_val_df = add_prefix_metadata(target_val_df)
        target_test_df = add_prefix_metadata(target_test_df)

        categorical_cols, numeric_cols = infer_column_types(source_train_df, aligned_feature_cols)
        temporal_numeric_cols = numeric_cols + [EVIDENCE_PROGRESS_COL, PREFIX_ROWS_SEEN_COL]
        assert_valid_hybrid_frame(source_train_df, aligned_feature_cols, categorical_cols, numeric_cols, "source_train", require_both_labels=True)
        assert_valid_hybrid_frame(target_val_df, aligned_feature_cols, categorical_cols, numeric_cols, "target_val", require_both_labels=False)
        assert_valid_hybrid_frame(target_test_df, aligned_feature_cols, categorical_cols, numeric_cols, "target_test", require_both_labels=False)

        artifacts = train_full_hybrid(
            source_df=source_train_df,
            categorical_cols=categorical_cols,
            numeric_cols=numeric_cols,
            temporal_numeric_cols=temporal_numeric_cols,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            token_dim=args.token_dim,
            transformer_depth=args.transformer_depth,
            transformer_heads=args.transformer_heads,
            tcn_hidden_dim=args.tcn_hidden_dim,
            tcn_blocks=args.tcn_blocks,
            prototype_embed_dim=args.prototype_embed_dim,
            gate_hidden_dim=args.gate_hidden_dim,
            dropout=args.dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            residual_dropout=residual_dropout,
            seed=args.seed,
            train_val_fraction=args.train_val_fraction,
            deterministic=args.deterministic,
            disable_tabular_branch=args.disable_tabular_branch,
            disable_temporal_branch=args.disable_temporal_branch,
            disable_prototype_branch=args.disable_prototype_branch,
            uniform_gating=args.uniform_gating,
        )

        artifact_payload = {
            "model_state_dict": artifacts.model.state_dict(),
            "preprocessor": artifacts.preprocessor,
            "categorical_cols": artifacts.categorical_cols,
            "numeric_cols": artifacts.numeric_cols,
            "temporal_numeric_cols": artifacts.temporal_numeric_cols,
            "seq_len": artifacts.seq_len,
            "device": str(DEVICE),
            "history": artifacts.history,
            "best_epoch": artifacts.best_epoch,
            "best_val_metrics": artifacts.best_val_metrics,
            "ablation_config": artifacts.ablation_config,
        }
        joblib.dump(artifact_payload, direction_dir / "hybrid_full_artifacts.joblib")
        joblib.dump(artifact_payload, direction_dir / "hybrid_full_best_checkpoint.joblib")
        pd.DataFrame(artifacts.history).to_csv(direction_dir / "training_history.csv", index=False)

        if target_dataset == "unsw":
            val_outputs = evaluate_full_hybrid_unsw_target_split("val", target_val_df, artifacts, fractions, direction_dir, args.eval_batch_size)
            test_outputs = evaluate_full_hybrid_unsw_target_split("test", target_test_df, artifacts, fractions, direction_dir, args.eval_batch_size)
            detail_key = "overall_unsw_attack_cat_summary.csv"
        else:
            val_outputs = evaluate_full_hybrid_iot23_target_split("val", target_val_df, artifacts, fractions, direction_dir, args.eval_batch_size)
            test_outputs = evaluate_full_hybrid_iot23_target_split("test", target_test_df, artifacts, fractions, direction_dir, args.eval_batch_size)
            detail_key = "overall_iot23_scenario_summary.csv"

        source_train_rows = int(len(source_train_df))
        target_val_rows = int(len(target_val_df))
        target_test_rows = int(len(target_test_df))
        direction_summary = pd.concat(
            [
                add_direction_metadata(val_outputs["summary"], direction_name, source_dataset, target_dataset, source_train_rows, target_val_rows, target_test_rows, len(aligned_feature_cols), ablation_name),
                add_direction_metadata(test_outputs["summary"], direction_name, source_dataset, target_dataset, source_train_rows, target_val_rows, target_test_rows, len(aligned_feature_cols), ablation_name),
            ],
            ignore_index=True,
        )
        direction_summary.to_csv(direction_dir / "overall_fraction_summary.csv", index=False)

        detail_df = pd.concat(
            [
                add_detail_metadata(val_outputs["details"], direction_name, "val", ablation_name),
                add_detail_metadata(test_outputs["details"], direction_name, "test", ablation_name),
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
            "deterministic": args.deterministic,
            "device": str(DEVICE),
            "ablation_name": ablation_name,
            "source_train_rows": source_train_rows,
            "target_val_rows_config": target_val_rows,
            "target_test_rows_config": target_test_rows,
            "aligned_features": aligned_feature_cols,
            "categorical_cols": categorical_cols,
            "numeric_cols": numeric_cols,
            "temporal_numeric_cols": temporal_numeric_cols,
            "seq_len": args.seq_len,
            "token_dim": args.token_dim,
            "transformer_depth": args.transformer_depth,
            "transformer_heads": args.transformer_heads,
            "tcn_hidden_dim": args.tcn_hidden_dim,
            "tcn_blocks": args.tcn_blocks,
            "prototype_embed_dim": args.prototype_embed_dim,
            "gate_hidden_dim": args.gate_hidden_dim,
            "dropout": args.dropout,
            "attention_dropout": attention_dropout,
            "ffn_dropout": ffn_dropout,
            "residual_dropout": residual_dropout,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "train_val_fraction": args.train_val_fraction,
            "disable_tabular_branch": args.disable_tabular_branch,
            "disable_temporal_branch": args.disable_temporal_branch,
            "disable_prototype_branch": args.disable_prototype_branch,
            "uniform_gating": args.uniform_gating,
            "best_epoch": artifacts.best_epoch,
            "best_val_metrics": artifacts.best_val_metrics,
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
