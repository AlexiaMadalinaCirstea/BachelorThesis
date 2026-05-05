from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

try:
    from .transfer_learning_early_detection_common import (
        DEFAULT_FRACTIONS,
        DEFAULT_TARGET_TRAIN_BUDGETS,
        budget_to_slug,
    )
except ImportError:
    THIS_DIR = Path(__file__).resolve().parent
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))
    from transfer_learning_early_detection_common import (
        DEFAULT_FRACTIONS,
        DEFAULT_TARGET_TRAIN_BUDGETS,
        budget_to_slug,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch matched multi-seed transfer-learning early-detection runs."
    )
    parser.add_argument(
        "--out_dir",
        default="early_detection/transfer_learning_based_early_detection/multiple_seeds_test",
        help="Parent directory for all run folders.",
    )
    parser.add_argument(
        "--directions",
        nargs="+",
        default=["iot23_to_unsw", "unsw_to_iot23"],
        choices=["iot23_to_unsw", "unsw_to_iot23"],
        help="Directions to run.",
    )
    parser.add_argument(
        "--target_train_budgets",
        nargs="+",
        type=int,
        default=DEFAULT_TARGET_TRAIN_BUDGETS,
        help="Target-train row budgets.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 456],
        help="Run seeds.",
    )
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=DEFAULT_FRACTIONS,
        help="Target evaluation fractions.",
    )
    parser.add_argument("--source_train_rows", type=int, default=100000, help="Source-train row cap.")
    parser.add_argument("--source_epochs", type=int, default=20, help="Source pretraining epochs.")
    parser.add_argument("--target_only_epochs", type=int, default=20, help="Target-only epochs.")
    parser.add_argument("--finetune_epochs", type=int, default=10, help="Transfer fine-tuning epochs.")
    parser.add_argument(
        "--overwrite_existing",
        action="store_true",
        help="Rerun folders even if they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_path = Path(__file__).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (script_path.parents[1] / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for direction in args.directions:
        for budget in args.target_train_budgets:
            budget_slug = budget_to_slug(int(budget))
            for seed in args.seeds:
                run_name = f"outputs_{direction}_budget{budget_slug}_seed{seed}"
                run_dir = out_dir / run_name
                if run_dir.exists() and not args.overwrite_existing:
                    print(f"[skip] {run_name}")
                    continue

                cmd = [
                    sys.executable,
                    str(script_path.parent / "run_transfer_learning_early_detection.py"),
                    "--direction",
                    direction,
                    "--out_dir",
                    str(run_dir),
                    "--target_train_rows",
                    str(int(budget)),
                    "--seed",
                    str(int(seed)),
                    "--source_train_rows",
                    str(int(args.source_train_rows)),
                    "--source_epochs",
                    str(int(args.source_epochs)),
                    "--target_only_epochs",
                    str(int(args.target_only_epochs)),
                    "--finetune_epochs",
                    str(int(args.finetune_epochs)),
                    "--fractions",
                    *[str(fraction) for fraction in args.fractions],
                ]
                print(f"[run] {run_name}")
                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
