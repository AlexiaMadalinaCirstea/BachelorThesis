from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from cross_domain_early_detection_common import DEFAULT_FRACTIONS


DEFAULT_SENSITIVITY_SEEDS = [42, 123, 456]
DEFAULT_BASELINE_CONFIG = {
    "iot_train_max_rows": 100000,
    "unsw_train_max_rows": 100000,
    "iot_eval_max_rows_per_scenario": 50000,
    "unsw_eval_max_rows": 30000,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run targeted size-sensitivity and eval-cap-sensitivity cross-domain early-detection experiments."
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
        default="early_detection/cross_domain_early_detection/sensitivity_tests",
        help="Directory where sensitivity run folders will be written.",
    )
    parser.add_argument(
        "--studies",
        nargs="+",
        choices=["size", "eval_cap"],
        default=["size", "eval_cap"],
        help="Sensitivity studies to run.",
    )
    parser.add_argument(
        "--directions",
        nargs="+",
        choices=["iot23_to_unsw", "unsw_to_iot23"],
        default=["iot23_to_unsw", "unsw_to_iot23"],
        help="Cross-domain transfer directions to run.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["rf", "mlp"],
        default=["rf", "mlp"],
        help="Model families to run.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SENSITIVITY_SEEDS,
        help="Seed subset for the targeted sensitivity study.",
    )
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=DEFAULT_FRACTIONS,
        help="Prefix fractions to evaluate.",
    )
    parser.add_argument(
        "--include_review_features",
        action="store_true",
        help="Include rows marked review_required in the aligned feature table.",
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
        default=DEFAULT_BASELINE_CONFIG["iot_train_max_rows"],
        help="Baseline IoT-23 source-train cap.",
    )
    parser.add_argument(
        "--unsw_train_max_rows",
        type=int,
        default=DEFAULT_BASELINE_CONFIG["unsw_train_max_rows"],
        help="Baseline UNSW-NB15 source-train cap.",
    )
    parser.add_argument(
        "--iot_eval_max_rows_per_scenario",
        type=int,
        default=DEFAULT_BASELINE_CONFIG["iot_eval_max_rows_per_scenario"],
        help="Baseline IoT-23 target eval cap per scenario.",
    )
    parser.add_argument(
        "--unsw_eval_max_rows",
        type=int,
        default=DEFAULT_BASELINE_CONFIG["unsw_eval_max_rows"],
        help="Baseline UNSW-NB15 target eval cap.",
    )
    parser.add_argument(
        "--iot_train_large_rows",
        type=int,
        default=200000,
        help="Larger IoT-23 source-train cap for the size-sensitivity study.",
    )
    parser.add_argument(
        "--unsw_train_large_rows",
        type=int,
        default=175341,
        help="Larger UNSW-NB15 source-train cap for the size-sensitivity study.",
    )
    parser.add_argument(
        "--iot_eval_large_rows_per_scenario",
        type=int,
        default=75000,
        help="Larger IoT-23 target eval cap per scenario for the eval-cap study.",
    )
    parser.add_argument(
        "--unsw_eval_large_rows",
        type=int,
        default=50000,
        help="Larger UNSW-NB15 target eval cap for the eval-cap study.",
    )
    parser.add_argument(
        "--rf_n_estimators",
        type=int,
        default=300,
        help="Number of Random Forest trees.",
    )
    parser.add_argument(
        "--rf_max_depth",
        type=int,
        default=None,
        help="Optional Random Forest max depth.",
    )
    parser.add_argument(
        "--rf_n_jobs",
        type=int,
        default=1,
        help="Random Forest parallel jobs.",
    )
    parser.add_argument(
        "--mlp_hidden_layers",
        nargs="+",
        type=int,
        default=[128, 64],
        help="Hidden layer sizes for the MLP runs.",
    )
    parser.add_argument(
        "--mlp_alpha",
        type=float,
        default=0.0001,
        help="MLP L2 penalty.",
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
    parser.add_argument(
        "--python_exe",
        default=sys.executable,
        help="Python interpreter used to launch the per-run scripts.",
    )
    parser.add_argument(
        "--overwrite_existing",
        action="store_true",
        help="Re-run even if the output folder already contains run_config.json.",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue with later runs if one sensitivity run fails.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the planned commands without executing them.",
    )
    return parser.parse_args()


def save_json(obj: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True)


def runner_path(thesis_root: Path, model: str) -> Path:
    filename = (
        "run_cross_domain_early_detection.py"
        if model == "rf"
        else "run_cross_domain_mlp_early_detection.py"
    )
    return thesis_root / "early_detection" / "cross_domain_early_detection" / filename


def build_run_name(direction: str, model: str, study: str, seed: int) -> str:
    return f"outputs_{direction}_{model}_{study}_seed{seed}"


def study_config_for_direction(args: argparse.Namespace, study: str, direction: str) -> dict[str, int]:
    config = {
        "iot_train_max_rows": args.iot_train_max_rows,
        "unsw_train_max_rows": args.unsw_train_max_rows,
        "iot_eval_max_rows_per_scenario": args.iot_eval_max_rows_per_scenario,
        "unsw_eval_max_rows": args.unsw_eval_max_rows,
    }
    if study == "size":
        if direction == "iot23_to_unsw":
            config["iot_train_max_rows"] = args.iot_train_large_rows
        else:
            config["unsw_train_max_rows"] = args.unsw_train_large_rows
    else:
        if direction == "iot23_to_unsw":
            config["unsw_eval_max_rows"] = args.unsw_eval_large_rows
        else:
            config["iot_eval_max_rows_per_scenario"] = args.iot_eval_large_rows_per_scenario
    return config


def build_command(
    thesis_root: Path,
    args: argparse.Namespace,
    direction: str,
    model: str,
    study: str,
    seed: int,
    out_dir: Path,
) -> tuple[list[str], dict[str, int]]:
    config = study_config_for_direction(args, study, direction)
    command = [
        args.python_exe,
        str(runner_path(thesis_root, model)),
        "--iot_data_dir",
        args.iot_data_dir,
        "--unsw_train_csv",
        args.unsw_train_csv,
        "--unsw_test_csv",
        args.unsw_test_csv,
        "--alignment_csv",
        args.alignment_csv,
        "--out_dir",
        str(out_dir),
        "--direction",
        direction,
        "--seed",
        str(seed),
        "--unsw_val_fraction",
        str(args.unsw_val_fraction),
        "--fractions",
        *[str(x) for x in args.fractions],
    ]

    if args.include_review_features:
        command.append("--include_review_features")

    if direction == "iot23_to_unsw":
        command.extend(["--iot_train_max_rows", str(config["iot_train_max_rows"])])
        command.extend(["--unsw_eval_max_rows", str(config["unsw_eval_max_rows"])])
    else:
        command.extend(["--unsw_train_max_rows", str(config["unsw_train_max_rows"])])
        command.extend(["--iot_eval_max_rows_per_scenario", str(config["iot_eval_max_rows_per_scenario"])])

    if model == "rf":
        command.extend(["--rf_n_estimators", str(args.rf_n_estimators)])
        if args.rf_max_depth is not None:
            command.extend(["--rf_max_depth", str(args.rf_max_depth)])
        command.extend(["--rf_n_jobs", str(args.rf_n_jobs)])
    else:
        command.extend(["--mlp_hidden_layers", *[str(x) for x in args.mlp_hidden_layers]])
        command.extend(["--mlp_alpha", str(args.mlp_alpha)])
        command.extend(["--mlp_max_iter", str(args.mlp_max_iter)])
        command.extend(["--mlp_batch_size", str(args.mlp_batch_size)])

    return command, config


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    thesis_root = script_dir.parents[1]

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (thesis_root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    planned_runs = []
    failures = []

    for study in args.studies:
        for direction in args.directions:
            for model in args.models:
                for seed in args.seeds:
                    run_name = build_run_name(direction, model, study, seed)
                    run_out_dir = out_dir / run_name
                    config_path = run_out_dir / "run_config.json"
                    command, config = build_command(
                        thesis_root=thesis_root,
                        args=args,
                        direction=direction,
                        model=model,
                        study=study,
                        seed=seed,
                        out_dir=run_out_dir,
                    )
                    planned_runs.append(
                        {
                            "run_name": run_name,
                            "study": study,
                            "direction": direction,
                            "model": model,
                            "seed": int(seed),
                            "out_dir": str(run_out_dir),
                            "effective_config": config,
                            "command": command,
                        }
                    )

                    if config_path.exists() and not args.overwrite_existing:
                        print(f"[skip] {run_name} already exists")
                        continue

                    run_out_dir.mkdir(parents=True, exist_ok=True)
                    print(f"[run] {run_name}")
                    print(" ".join(command))
                    if args.dry_run:
                        continue

                    try:
                        subprocess.run(command, cwd=thesis_root, check=True)
                    except subprocess.CalledProcessError as exc:
                        failures.append(
                            {
                                "run_name": run_name,
                                "study": study,
                                "direction": direction,
                                "model": model,
                                "seed": int(seed),
                                "returncode": int(exc.returncode),
                            }
                        )
                        if not args.continue_on_error:
                            save_json(
                                {
                                    "baseline_config": DEFAULT_BASELINE_CONFIG,
                                    "planned_runs": planned_runs,
                                    "failures": failures,
                                },
                                out_dir / "sensitivity_run_manifest.json",
                            )
                            raise

    save_json(
        {
            "python_exe": args.python_exe,
            "runs_dir": str(out_dir),
            "baseline_config": DEFAULT_BASELINE_CONFIG,
            "seeds": [int(x) for x in args.seeds],
            "studies": list(args.studies),
            "directions": list(args.directions),
            "models": list(args.models),
            "fractions": [float(x) for x in args.fractions],
            "planned_run_count": len(planned_runs),
            "failure_count": len(failures),
            "planned_runs": planned_runs,
            "failures": failures,
        },
        out_dir / "sensitivity_run_manifest.json",
    )

    if failures:
        raise SystemExit(f"Completed with {len(failures)} failed runs. See sensitivity_run_manifest.json.")


if __name__ == "__main__":
    main()
