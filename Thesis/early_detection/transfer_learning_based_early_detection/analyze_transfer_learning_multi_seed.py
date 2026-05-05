from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUN_PATTERN = re.compile(r"^outputs_(iot23_to_unsw|unsw_to_iot23)_budget([A-Za-z0-9]+)_seed(\d+)$")
PRIMARY_METRICS = ["f1_attack", "recall_attack", "f1_macro", "accuracy"]
TRANSFER_CONDITIONS = ["source_only", "target_only", "transfer_adapted"]
LOW_FRACTION_MAX = 0.20
DEFAULT_GAIN_EPSILON = 0.005
T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}
DIRECTION_LABELS = {
    "iot23_to_unsw": "IoT-23 -> UNSW-NB15",
    "unsw_to_iot23": "UNSW-NB15 -> IoT-23",
}
CONDITION_LABELS = {
    "source_only": "Source-only",
    "target_only": "Target-only",
    "transfer_adapted": "Transfer-adapted",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate repeated transfer-learning early-detection runs."
    )
    parser.add_argument(
        "--runs_dir",
        default="early_detection/transfer_learning_based_early_detection/multiple_seeds_test",
        help="Directory containing outputs_<direction>_budget<budget>_seed<seed> folders.",
    )
    parser.add_argument(
        "--out_dir",
        default="early_detection/transfer_learning_based_early_detection/multi_seed_analyzer",
        help="Directory for aggregated outputs.",
    )
    parser.add_argument(
        "--gain_epsilon",
        type=float,
        default=DEFAULT_GAIN_EPSILON,
        help="Neutrality margin for transfer gain classification.",
    )
    return parser.parse_args()


def save_json(obj: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True)


def parse_run_name(name: str) -> dict | None:
    match = RUN_PATTERN.match(name)
    if not match:
        return None
    direction, budget_slug, seed = match.groups()
    return {
        "direction": direction,
        "budget_slug": budget_slug,
        "seed": int(seed),
        "run_name": name,
    }


def t_critical_95(n: int) -> float:
    if n <= 1:
        return math.nan
    return T_CRITICAL_95.get(n - 1, 1.96)


def summarize_metric(values: pd.Series) -> dict[str, float]:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    n = int(arr.size)
    if n == 0:
        return {
            "n_seeds": 0,
            "mean": math.nan,
            "std": math.nan,
            "se": math.nan,
            "ci95_low": math.nan,
            "ci95_high": math.nan,
        }
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    se = float(std / math.sqrt(n)) if n > 1 else 0.0
    margin = float(t_critical_95(n) * se) if n > 1 else 0.0
    return {
        "n_seeds": n,
        "mean": mean,
        "std": std,
        "se": se,
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
    }


def exact_sign_test_two_sided(deltas: np.ndarray) -> tuple[int, int, int, float]:
    deltas = deltas[np.isfinite(deltas)]
    positive = int((deltas > 0).sum())
    negative = int((deltas < 0).sum())
    nonzero = positive + negative
    if nonzero == 0:
        return positive, negative, nonzero, 1.0
    smaller = min(positive, negative)
    cdf = sum(math.comb(nonzero, k) for k in range(smaller + 1)) / (2**nonzero)
    return positive, negative, nonzero, float(min(1.0, 2.0 * cdf))


def trapezoid_auc(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def collect_runs(runs_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_rows = []
    inventory_rows = []

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        meta = parse_run_name(run_dir.name)
        if meta is None:
            continue

        config_path = run_dir / "run_config.json"
        summary_path = run_dir / "overall_fraction_summary.csv"
        if not config_path.exists() or not summary_path.exists():
            raise FileNotFoundError(f"Missing required files in {run_dir}")

        with open(config_path, "r", encoding="utf-8") as handle:
            run_config = json.load(handle)
        summary_df = pd.read_csv(summary_path)
        summary_df = summary_df[summary_df["split"] == "test"].copy()

        summary_df["direction"] = meta["direction"]
        summary_df["budget_slug"] = meta["budget_slug"]
        summary_df["seed"] = meta["seed"]
        summary_df["run_name"] = meta["run_name"]
        summary_df["direction_label"] = summary_df["direction"].map(DIRECTION_LABELS)
        summary_df["condition_label"] = summary_df["condition"].map(CONDITION_LABELS)
        summary_df["target_train_rows_config"] = run_config.get("target_train_rows")
        summary_df["source_train_rows_config"] = run_config.get("source_train_rows")
        summary_df["target_val_rows_config"] = run_config.get("target_val_rows")
        summary_df["target_test_rows_config"] = run_config.get("target_test_rows")

        run_rows.append(summary_df)
        inventory_rows.append(
            {
                **meta,
                "target_train_rows_config": run_config.get("target_train_rows"),
                "source_train_rows_config": run_config.get("source_train_rows"),
                "fraction_count": int(summary_df["fraction"].nunique()),
                "conditions": ",".join(sorted(summary_df["condition"].unique().tolist())),
            }
        )

    if not run_rows:
        raise AssertionError(f"No runs found in {runs_dir}")

    combined_df = pd.concat(run_rows, ignore_index=True)
    inventory_df = pd.DataFrame(inventory_rows).sort_values(["direction", "target_train_rows_config", "seed"]).reset_index(drop=True)
    return combined_df, inventory_df


def aggregate_conditions(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = summary_df.groupby(["direction", "target_train_rows_config", "condition", "fraction"], sort=True)
    for (direction, budget, condition, fraction), group in grouped:
        row = {
            "direction": direction,
            "direction_label": group["direction_label"].iloc[0],
            "target_train_rows_config": int(budget),
            "condition": condition,
            "condition_label": group["condition_label"].iloc[0],
            "fraction": float(fraction),
        }
        for metric in PRIMARY_METRICS:
            stats = summarize_metric(group[metric])
            for key, value in stats.items():
                row[f"{metric}_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["direction", "target_train_rows_config", "condition", "fraction"]).reset_index(drop=True)


def build_transfer_gain_rows(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pivot_df = summary_df.pivot_table(
        index=["direction", "target_train_rows_config", "seed", "fraction"],
        columns="condition",
        values=PRIMARY_METRICS,
        aggfunc="first",
    )
    for (direction, budget, fraction), group in pivot_df.groupby(level=["direction", "target_train_rows_config", "fraction"]):
        row = {
            "direction": direction,
            "direction_label": DIRECTION_LABELS[direction],
            "target_train_rows_config": int(budget),
            "fraction": float(fraction),
        }
        for metric in PRIMARY_METRICS:
            transfer = group[(metric, "transfer_adapted")].to_numpy(dtype=float)
            target_only = group[(metric, "target_only")].to_numpy(dtype=float)
            source_only = group[(metric, "source_only")].to_numpy(dtype=float)
            gain_vs_target = transfer - target_only
            gain_vs_source = transfer - source_only

            target_stats = summarize_metric(pd.Series(gain_vs_target))
            source_stats = summarize_metric(pd.Series(gain_vs_source))
            pos_t, neg_t, nonzero_t, p_t = exact_sign_test_two_sided(gain_vs_target)
            pos_s, neg_s, nonzero_s, p_s = exact_sign_test_two_sided(gain_vs_source)

            for key, value in target_stats.items():
                row[f"{metric}_gain_transfer_minus_target_only_{key}"] = value
            row[f"{metric}_gain_transfer_minus_target_only_positive_seeds"] = pos_t
            row[f"{metric}_gain_transfer_minus_target_only_negative_seeds"] = neg_t
            row[f"{metric}_gain_transfer_minus_target_only_nonzero_seeds"] = nonzero_t
            row[f"{metric}_gain_transfer_minus_target_only_sign_test_pvalue"] = p_t

            for key, value in source_stats.items():
                row[f"{metric}_gain_transfer_minus_source_only_{key}"] = value
            row[f"{metric}_gain_transfer_minus_source_only_positive_seeds"] = pos_s
            row[f"{metric}_gain_transfer_minus_source_only_negative_seeds"] = neg_s
            row[f"{metric}_gain_transfer_minus_source_only_nonzero_seeds"] = nonzero_s
            row[f"{metric}_gain_transfer_minus_source_only_sign_test_pvalue"] = p_s
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["direction", "target_train_rows_config", "fraction"]).reset_index(drop=True)


def build_curve_level_seed_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (direction, budget, condition, seed), group in summary_df.groupby(
        ["direction", "target_train_rows_config", "condition", "seed"],
        sort=True,
    ):
        ordered = group.sort_values("fraction")
        x = ordered["fraction"].to_numpy(dtype=float)
        span = float(x[-1] - x[0]) if len(x) > 1 else 0.0
        row = {
            "direction": direction,
            "direction_label": DIRECTION_LABELS[direction],
            "target_train_rows_config": int(budget),
            "condition": condition,
            "condition_label": CONDITION_LABELS[condition],
            "seed": int(seed),
        }
        for metric in PRIMARY_METRICS:
            y = ordered[metric].to_numpy(dtype=float)
            auc = trapezoid_auc(y, x)
            low_df = ordered[ordered["fraction"] <= LOW_FRACTION_MAX]
            row[f"auc_{metric}"] = auc
            row[f"auc_{metric}_normalized"] = float(auc / span) if span > 0 else float(y[0])
            row[f"low_fraction_mean_{metric}"] = float(low_df[metric].mean())
            row[f"full_fraction_{metric}"] = float(ordered.iloc[-1][metric])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["direction", "target_train_rows_config", "condition", "seed"]).reset_index(drop=True)


def build_curve_level_gain_summary(curve_df: pd.DataFrame, gain_epsilon: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    pivot_df = curve_df.pivot_table(
        index=["direction", "target_train_rows_config", "seed"],
        columns="condition",
        values=[col for col in curve_df.columns if col.startswith("auc_") or col.startswith("low_fraction_mean_") or col.startswith("full_fraction_")],
        aggfunc="first",
    )
    rows = []
    class_rows = []
    metrics = [
        "auc_f1_attack_normalized",
        "auc_recall_attack_normalized",
        "auc_f1_macro_normalized",
        "auc_accuracy_normalized",
        "low_fraction_mean_f1_attack",
        "full_fraction_f1_attack",
        "full_fraction_recall_attack",
    ]
    for (direction, budget), group in pivot_df.groupby(level=["direction", "target_train_rows_config"]):
        row = {
            "direction": direction,
            "direction_label": DIRECTION_LABELS[direction],
            "target_train_rows_config": int(budget),
        }
        for metric in metrics:
            gain = group[(metric, "transfer_adapted")].to_numpy(dtype=float) - group[(metric, "target_only")].to_numpy(dtype=float)
            stats = summarize_metric(pd.Series(gain))
            pos, neg, nonzero, p_value = exact_sign_test_two_sided(gain)
            for key, value in stats.items():
                row[f"{metric}_gain_transfer_minus_target_only_{key}"] = value
            row[f"{metric}_gain_transfer_minus_target_only_positive_seeds"] = pos
            row[f"{metric}_gain_transfer_minus_target_only_negative_seeds"] = neg
            row[f"{metric}_gain_transfer_minus_target_only_nonzero_seeds"] = nonzero
            row[f"{metric}_gain_transfer_minus_target_only_sign_test_pvalue"] = p_value

            for seed_index, gain_value in enumerate(gain):
                if metric != "full_fraction_f1_attack":
                    continue
                if gain_value > gain_epsilon:
                    label = "positive"
                elif gain_value < -gain_epsilon:
                    label = "negative"
                else:
                    label = "neutral"
                class_rows.append(
                    {
                        "direction": direction,
                        "target_train_rows_config": int(budget),
                        "seed_order": int(seed_index),
                        "metric": metric,
                        "gain_value": float(gain_value),
                        "classification": label,
                    }
                )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["direction", "target_train_rows_config"]).reset_index(drop=True), pd.DataFrame(class_rows)


def build_fraction_gain_classification(gain_df: pd.DataFrame, gain_epsilon: float) -> pd.DataFrame:
    rows = []
    for _, row in gain_df.iterrows():
        gain_value = float(row["f1_attack_gain_transfer_minus_target_only_mean"])
        if gain_value > gain_epsilon:
            label = "positive"
        elif gain_value < -gain_epsilon:
            label = "negative"
        else:
            label = "neutral"
        rows.append(
            {
                "direction": row["direction"],
                "direction_label": row["direction_label"],
                "target_train_rows_config": int(row["target_train_rows_config"]),
                "fraction": float(row["fraction"]),
                "mean_gain_f1_attack": gain_value,
                "classification": label,
            }
        )
    return pd.DataFrame(rows).sort_values(["direction", "target_train_rows_config", "fraction"]).reset_index(drop=True)


def plot_condition_curves(condition_df: pd.DataFrame, out_dir: Path) -> None:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "source_only": "#F58518",
        "target_only": "#54A24B",
        "transfer_adapted": "#4C78A8",
    }
    for direction in sorted(condition_df["direction"].unique()):
        for budget in sorted(condition_df["target_train_rows_config"].unique()):
            subset = condition_df[
                (condition_df["direction"] == direction) & (condition_df["target_train_rows_config"] == budget)
            ].copy()
            if subset.empty:
                continue
            plt.figure(figsize=(8.5, 5.5))
            for condition in TRANSFER_CONDITIONS:
                cond_df = subset[subset["condition"] == condition].sort_values("fraction")
                if cond_df.empty:
                    continue
                x = cond_df["fraction"].to_numpy(dtype=float)
                mean = cond_df["f1_attack_mean"].to_numpy(dtype=float)
                low = cond_df["f1_attack_ci95_low"].to_numpy(dtype=float)
                high = cond_df["f1_attack_ci95_high"].to_numpy(dtype=float)
                plt.plot(x, mean, marker="o", linewidth=2, color=colors[condition], label=CONDITION_LABELS[condition])
                plt.fill_between(x, low, high, color=colors[condition], alpha=0.18)
            plt.title(f"{DIRECTION_LABELS[direction]} | target train rows = {budget}")
            plt.xlabel("Target Prefix Fraction")
            plt.ylabel("Attack F1")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(plots_dir / f"{direction}_budget{budget}_f1_attack_conditions.png", dpi=200)
            plt.close()


def plot_gain_curves(gain_df: pd.DataFrame, out_dir: Path) -> None:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    palette = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]

    for direction in sorted(gain_df["direction"].unique()):
        subset = gain_df[gain_df["direction"] == direction].copy()
        plt.figure(figsize=(8.5, 5.5))
        plt.axhline(0.0, color="black", linewidth=1, alpha=0.5)
        for idx, budget in enumerate(sorted(subset["target_train_rows_config"].unique())):
            budget_df = subset[subset["target_train_rows_config"] == budget].sort_values("fraction")
            x = budget_df["fraction"].to_numpy(dtype=float)
            mean = budget_df["f1_attack_gain_transfer_minus_target_only_mean"].to_numpy(dtype=float)
            low = budget_df["f1_attack_gain_transfer_minus_target_only_ci95_low"].to_numpy(dtype=float)
            high = budget_df["f1_attack_gain_transfer_minus_target_only_ci95_high"].to_numpy(dtype=float)
            color = palette[idx % len(palette)]
            plt.plot(x, mean, marker="o", linewidth=2, color=color, label=f"target train rows = {budget}")
            plt.fill_between(x, low, high, color=color, alpha=0.18)
        plt.title(f"Transfer Gain over Target-only | {DIRECTION_LABELS[direction]}")
        plt.xlabel("Target Prefix Fraction")
        plt.ylabel("Attack F1 gain")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / f"{direction}_f1_attack_gain_transfer_minus_target_only.png", dpi=200)
        plt.close()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_absolute():
        runs_dir = (script_dir.parents[1] / runs_dir).resolve()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (script_dir.parents[1] / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_df, inventory_df = collect_runs(runs_dir)
    condition_stats_df = aggregate_conditions(summary_df)
    gain_stats_df = build_transfer_gain_rows(summary_df)
    curve_seed_df = build_curve_level_seed_summary(summary_df)
    curve_gain_df, curve_gain_class_df = build_curve_level_gain_summary(curve_seed_df, args.gain_epsilon)
    fraction_gain_class_df = build_fraction_gain_classification(gain_stats_df, args.gain_epsilon)

    inventory_df.to_csv(out_dir / "run_inventory.csv", index=False)
    summary_df.sort_values(["direction", "target_train_rows_config", "condition", "seed", "fraction"]).to_csv(
        out_dir / "all_seed_test_rows.csv",
        index=False,
    )
    condition_stats_df.to_csv(out_dir / "per_condition_fraction_summary_stats.csv", index=False)
    gain_stats_df.to_csv(out_dir / "per_fraction_transfer_gain_stats.csv", index=False)
    curve_seed_df.to_csv(out_dir / "curve_level_seed_summary.csv", index=False)
    curve_gain_df.to_csv(out_dir / "curve_level_transfer_gain_stats.csv", index=False)
    curve_gain_class_df.to_csv(out_dir / "curve_level_transfer_gain_classification.csv", index=False)
    fraction_gain_class_df.to_csv(out_dir / "fraction_level_transfer_gain_classification.csv", index=False)

    plot_condition_curves(condition_stats_df, out_dir)
    plot_gain_curves(gain_stats_df, out_dir)

    save_json(
        {
            "runs_dir": str(runs_dir),
            "out_dir": str(out_dir),
            "gain_epsilon": args.gain_epsilon,
            "n_total_runs": int(inventory_df.shape[0]),
            "n_unique_seeds": int(inventory_df["seed"].nunique()),
            "directions": sorted(inventory_df["direction"].unique().tolist()),
            "target_train_rows": sorted(inventory_df["target_train_rows_config"].unique().tolist()),
        },
        out_dir / "analysis_manifest.json",
    )


if __name__ == "__main__":
    main()
