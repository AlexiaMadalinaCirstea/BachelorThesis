from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the updated transfer-learning recipe for multiple random seeds "
            "and aggregate the results into seed-stability tables."
        )
    )
    parser.add_argument(
        "--python_exe",
        default=sys.executable,
        help="Python executable used to launch transfer_learning_updated_recipe.py.",
    )
    parser.add_argument(
        "--updated_recipe_script",
        default="transfer_learning/transfer_learning_updated_recipe.py",
        help="Path to transfer_learning_updated_recipe.py.",
    )
    parser.add_argument(
        "--base_out_dir",
        default="transfer_learning/seed_stability",
        help="Base output directory for per-seed runs and aggregate summaries.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 2026],
        help="Random seeds to evaluate.",
    )
    parser.add_argument(
        "--target_fractions",
        nargs="+",
        type=float,
        default=[0.05, 0.10, 0.25, 0.50, 1.0],
        help="Target fractions passed through to the updated recipe.",
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
    parser.add_argument("--iot_train_max_rows", type=int, default=150000)
    parser.add_argument("--iot_test_max_rows", type=int, default=50000)
    parser.add_argument("--unsw_train_max_rows", type=int, default=150000)
    parser.add_argument("--unsw_test_max_rows", type=int, default=30000)
    parser.add_argument("--balance_target_train", action="store_true")
    parser.add_argument("--target_balance_ratio", type=float, default=1.0)
    parser.add_argument("--calibration_fraction", type=float, default=0.20)
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
    )
    parser.add_argument("--pretrain_estimators", type=int, default=50)
    parser.add_argument("--adapt_estimators", type=int, default=150)
    parser.add_argument("--target_only_estimators", type=int, default=150)
    parser.add_argument("--xgb_max_depth", type=int, default=6)
    parser.add_argument("--xgb_learning_rate", type=float, default=0.05)
    return parser.parse_args()


def run_one_seed(args: argparse.Namespace, seed: int, out_dir: Path) -> Path:
    cmd = [
        args.python_exe,
        args.updated_recipe_script,
        "--iot_train",
        args.iot_train,
        "--iot_test",
        args.iot_test,
        "--unsw_train",
        args.unsw_train,
        "--unsw_test",
        args.unsw_test,
        "--alignment_csv",
        args.alignment_csv,
        "--out_dir",
        str(out_dir),
        "--target_fractions",
        *[str(x) for x in args.target_fractions],
        "--iot_train_max_rows",
        str(args.iot_train_max_rows),
        "--iot_test_max_rows",
        str(args.iot_test_max_rows),
        "--unsw_train_max_rows",
        str(args.unsw_train_max_rows),
        "--unsw_test_max_rows",
        str(args.unsw_test_max_rows),
        "--target_balance_ratio",
        str(args.target_balance_ratio),
        "--calibration_fraction",
        str(args.calibration_fraction),
        "--thresholds",
        *[str(x) for x in args.thresholds],
        "--pretrain_estimators",
        str(args.pretrain_estimators),
        "--adapt_estimators",
        str(args.adapt_estimators),
        "--target_only_estimators",
        str(args.target_only_estimators),
        "--xgb_max_depth",
        str(args.xgb_max_depth),
        "--xgb_learning_rate",
        str(args.xgb_learning_rate),
        "--random_state",
        str(seed),
    ]
    if args.balance_target_train:
        cmd.append("--balance_target_train")

    subprocess.run(cmd, check=True)
    return out_dir / "updated_recipe_summary.csv"


def aggregate_seed_results(summary_paths: list[tuple[int, Path]], out_dir: Path) -> None:
    frames = []
    for seed, path in summary_paths:
        df = pd.read_csv(path)
        df["seed"] = seed
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(out_dir / "combined_seed_results.csv", index=False)

    grouped = (
        combined.groupby(["direction", "condition", "target_fraction"], as_index=False)
        .agg(
            mean_f1_macro=("f1_macro", "mean"),
            std_f1_macro=("f1_macro", "std"),
            min_f1_macro=("f1_macro", "min"),
            max_f1_macro=("f1_macro", "max"),
            mean_f1_attack=("f1_attack", "mean"),
            std_f1_attack=("f1_attack", "std"),
            mean_recall_attack=("recall_attack", "mean"),
            std_recall_attack=("recall_attack", "std"),
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
        )
        .sort_values(["direction", "condition", "target_fraction"])
    )
    grouped.to_csv(out_dir / "seed_stability_summary.csv", index=False)

    target_only = grouped[grouped["condition"] == "target_only_updated"][
        ["direction", "target_fraction", "mean_f1_macro"]
    ].rename(columns={"mean_f1_macro": "target_only_mean_f1_macro"})
    transfer = grouped[grouped["condition"] == "transfer_learning_updated"].copy()
    gain = transfer.merge(target_only, on=["direction", "target_fraction"], how="left")
    gain["mean_macro_f1_gain_vs_target_only"] = (
        gain["mean_f1_macro"] - gain["target_only_mean_f1_macro"]
    )
    gain[
        [
            "direction",
            "target_fraction",
            "mean_f1_macro",
            "std_f1_macro",
            "target_only_mean_f1_macro",
            "mean_macro_f1_gain_vs_target_only",
        ]
    ].to_csv(out_dir / "seed_gain_vs_target_only.csv", index=False)


def main() -> None:
    args = parse_args()
    base_out_dir = Path(args.base_out_dir)
    base_out_dir.mkdir(parents=True, exist_ok=True)

    summary_paths: list[tuple[int, Path]] = []
    for seed in args.seeds:
        seed_out_dir = base_out_dir / f"seed_{seed}"
        seed_out_dir.mkdir(parents=True, exist_ok=True)
        summary_path = run_one_seed(args=args, seed=seed, out_dir=seed_out_dir)
        summary_paths.append((seed, summary_path))

    aggregate_seed_results(summary_paths=summary_paths, out_dir=base_out_dir)

    with open(base_out_dir / "seed_run_config.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "seeds": args.seeds,
                "target_fractions": args.target_fractions,
                "balance_target_train": args.balance_target_train,
                "target_balance_ratio": args.target_balance_ratio,
                "calibration_fraction": args.calibration_fraction,
                "pretrain_estimators": args.pretrain_estimators,
                "adapt_estimators": args.adapt_estimators,
                "target_only_estimators": args.target_only_estimators,
                "xgb_max_depth": args.xgb_max_depth,
                "xgb_learning_rate": args.xgb_learning_rate,
            },
            handle,
            indent=2,
        )

    print("Completed multi-seed transfer-learning stability runs.")
    print(f"Saved outputs to: {base_out_dir}")


if __name__ == "__main__":
    main()
