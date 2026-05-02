from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUN_PATTERN = re.compile(r"^outputs_(iot23_to_unsw|unsw_to_iot23)_(rf|mlp)_seed(\d+)$")
METRICS = ["f1_attack", "recall_attack", "f1_macro", "accuracy"]
LOW_FRACTION_MAX = 0.20
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
MODEL_LABELS = {
    "rf": "RF",
    "mlp": "MLP",
}


def trapezoid_auc(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate multi-seed cross-domain early-detection runs into summary statistics and plots."
    )
    parser.add_argument(
        "--runs_dir",
        default="early_detection/cross_domain_early_detection/multiple_seeds_test",
        help="Directory containing outputs_<direction>_<model>_seed<seed> runs.",
    )
    parser.add_argument(
        "--out_dir",
        default="early_detection/cross_domain_early_detection/cross_domain_multi_seed_analyzer",
        help="Directory for aggregated multi-seed outputs.",
    )
    return parser.parse_args()


def save_json(obj: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True)


def parse_run_name(name: str) -> dict | None:
    match = RUN_PATTERN.match(name)
    if not match:
        return None
    direction, model, seed = match.groups()
    return {
        "direction": direction,
        "model": model,
        "seed": int(seed),
        "run_name": name,
    }


def read_run_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_source_train_rows(run_config: dict) -> int | None:
    return run_config.get("iot_train_rows") or run_config.get("unsw_source_train_rows")


def extract_target_val_rows(run_config: dict) -> int | None:
    return run_config.get("unsw_target_val_rows") or run_config.get("iot_val_rows")


def extract_target_test_rows(run_config: dict) -> int | None:
    return run_config.get("unsw_test_rows") or run_config.get("iot_test_rows")


def load_overall_fraction_summary(run_dir: Path, run_meta: dict, run_config: dict) -> pd.DataFrame:
    summary_path = run_dir / "overall_fraction_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")

    df = pd.read_csv(summary_path)
    df = df[df["split"] == "test"].copy()
    if df.empty:
        raise AssertionError(f"No test rows found in {summary_path}")

    for key, value in run_meta.items():
        df[key] = value

    df["direction_label"] = df["direction"].map(DIRECTION_LABELS)
    df["model_label"] = df["model"].map(MODEL_LABELS)
    df["source_train_rows_config"] = extract_source_train_rows(run_config)
    df["target_val_rows_config"] = extract_target_val_rows(run_config)
    df["target_test_rows_config"] = extract_target_test_rows(run_config)
    df["config_seed"] = run_config.get("seed")
    df["n_aligned_features_config"] = run_config.get("n_aligned_features")
    return df


def collect_run_data(runs_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_rows = []
    inventory_rows = []

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        run_meta = parse_run_name(run_dir.name)
        if run_meta is None:
            continue

        config_path = run_dir / "run_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing run_config.json in {run_dir}")
        run_config = read_run_config(config_path)

        summary_df = load_overall_fraction_summary(run_dir, run_meta, run_config)
        all_rows.append(summary_df)
        inventory_rows.append(
            {
                **run_meta,
                "source_train_rows_config": extract_source_train_rows(run_config),
                "target_val_rows_config": extract_target_val_rows(run_config),
                "target_test_rows_config": extract_target_test_rows(run_config),
                "n_aligned_features_config": run_config.get("n_aligned_features"),
                "fractions": ",".join(str(x) for x in sorted(summary_df["fraction"].unique().tolist())),
                "fraction_count": int(summary_df["fraction"].nunique()),
            }
        )

    if not all_rows:
        raise AssertionError(f"No multi-seed runs found in {runs_dir}")

    combined_df = pd.concat(all_rows, ignore_index=True)
    inventory_df = pd.DataFrame(inventory_rows).sort_values(["direction", "model", "seed"]).reset_index(drop=True)
    return combined_df, inventory_df


def t_critical_95(n: int) -> float:
    if n <= 1:
        return math.nan
    df = n - 1
    if df in T_CRITICAL_95:
        return T_CRITICAL_95[df]
    return 1.96


def summarize_metric(series: pd.Series) -> dict:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    n = int(values.size)
    if n == 0:
        return {
            "n_seeds": 0,
            "mean": math.nan,
            "std": math.nan,
            "se": math.nan,
            "ci95_low": math.nan,
            "ci95_high": math.nan,
        }

    mean = float(values.mean())
    std = float(values.std(ddof=1)) if n > 1 else 0.0
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


def aggregate_per_fraction(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = summary_df.groupby(["direction", "model", "fraction"], sort=True)

    for (direction, model, fraction), group in grouped:
        row = {
            "direction": direction,
            "model": model,
            "direction_label": group["direction_label"].iloc[0],
            "model_label": group["model_label"].iloc[0],
            "fraction": float(fraction),
            "source_train_rows_config": int(group["source_train_rows_config"].iloc[0]),
            "target_val_rows_config": int(group["target_val_rows_config"].iloc[0]),
            "target_test_rows_config": int(group["target_test_rows_config"].iloc[0]),
            "n_aligned_features_config": int(group["n_aligned_features_config"].iloc[0]),
        }
        for metric in METRICS:
            stats = summarize_metric(group[metric])
            for key, value in stats.items():
                row[f"{metric}_{key}"] = value
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["direction", "model", "fraction"]).reset_index(drop=True)


def build_curve_level_seed_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (direction, model, seed), group in summary_df.groupby(["direction", "model", "seed"], sort=True):
        ordered = group.sort_values("fraction")
        fractions = ordered["fraction"].to_numpy(dtype=float)
        f1_attack = ordered["f1_attack"].to_numpy(dtype=float)
        recall_attack = ordered["recall_attack"].to_numpy(dtype=float)
        f1_macro = ordered["f1_macro"].to_numpy(dtype=float)
        accuracy = ordered["accuracy"].to_numpy(dtype=float)

        span = float(fractions[-1] - fractions[0]) if len(fractions) > 1 else 0.0
        auc_f1_attack = trapezoid_auc(f1_attack, fractions)
        auc_recall_attack = trapezoid_auc(recall_attack, fractions)
        auc_f1_macro = trapezoid_auc(f1_macro, fractions)
        auc_accuracy = trapezoid_auc(accuracy, fractions)

        low_fraction_df = ordered[ordered["fraction"] <= LOW_FRACTION_MAX]
        full_fraction_row = ordered.loc[ordered["fraction"] == ordered["fraction"].max()].iloc[0]
        rows.append(
            {
                "direction": direction,
                "model": model,
                "seed": int(seed),
                "direction_label": ordered["direction_label"].iloc[0],
                "model_label": ordered["model_label"].iloc[0],
                "auc_f1_attack": auc_f1_attack,
                "auc_f1_attack_normalized": float(auc_f1_attack / span) if span > 0 else float(f1_attack[0]),
                "auc_recall_attack": auc_recall_attack,
                "auc_recall_attack_normalized": float(auc_recall_attack / span) if span > 0 else float(recall_attack[0]),
                "auc_f1_macro": auc_f1_macro,
                "auc_f1_macro_normalized": float(auc_f1_macro / span) if span > 0 else float(f1_macro[0]),
                "auc_accuracy": auc_accuracy,
                "auc_accuracy_normalized": float(auc_accuracy / span) if span > 0 else float(accuracy[0]),
                "low_fraction_mean_f1_attack": float(low_fraction_df["f1_attack"].mean()),
                "low_fraction_mean_recall_attack": float(low_fraction_df["recall_attack"].mean()),
                "low_fraction_mean_f1_macro": float(low_fraction_df["f1_macro"].mean()),
                "low_fraction_mean_accuracy": float(low_fraction_df["accuracy"].mean()),
                "full_fraction_f1_attack": float(full_fraction_row["f1_attack"]),
                "full_fraction_recall_attack": float(full_fraction_row["recall_attack"]),
                "full_fraction_f1_macro": float(full_fraction_row["f1_macro"]),
                "full_fraction_accuracy": float(full_fraction_row["accuracy"]),
            }
        )

    return pd.DataFrame(rows).sort_values(["direction", "model", "seed"]).reset_index(drop=True)


def aggregate_curve_level(curve_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = [
        "auc_f1_attack",
        "auc_f1_attack_normalized",
        "auc_recall_attack",
        "auc_recall_attack_normalized",
        "auc_f1_macro",
        "auc_f1_macro_normalized",
        "auc_accuracy",
        "auc_accuracy_normalized",
        "low_fraction_mean_f1_attack",
        "low_fraction_mean_recall_attack",
        "low_fraction_mean_f1_macro",
        "low_fraction_mean_accuracy",
        "full_fraction_f1_attack",
        "full_fraction_recall_attack",
        "full_fraction_f1_macro",
        "full_fraction_accuracy",
    ]

    for (direction, model), group in curve_df.groupby(["direction", "model"], sort=True):
        row = {
            "direction": direction,
            "model": model,
            "direction_label": group["direction_label"].iloc[0],
            "model_label": group["model_label"].iloc[0],
        }
        for metric in metrics:
            stats = summarize_metric(group[metric])
            for key, value in stats.items():
                row[f"{metric}_{key}"] = value
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["direction", "model"]).reset_index(drop=True)


def exact_sign_test_two_sided(deltas: np.ndarray) -> tuple[int, int, int, float]:
    deltas = deltas[np.isfinite(deltas)]
    positive = int((deltas > 0).sum())
    negative = int((deltas < 0).sum())
    nonzero = positive + negative

    if nonzero == 0:
        return positive, negative, nonzero, 1.0

    smaller = min(positive, negative)
    cdf = sum(math.comb(nonzero, k) for k in range(smaller + 1)) / (2**nonzero)
    p_value = min(1.0, 2.0 * cdf)
    return positive, negative, nonzero, float(p_value)


def build_paired_fraction_deltas(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pivot_df = summary_df.pivot_table(
        index=["direction", "seed", "fraction"],
        columns="model",
        values=METRICS,
        aggfunc="first",
    )
    if pivot_df.empty:
        return pd.DataFrame()

    available_models = set(pivot_df.columns.get_level_values(1))
    if not {"rf", "mlp"}.issubset(available_models):
        raise AssertionError("Paired multi-seed comparison requires both rf and mlp runs.")

    for (direction, fraction), group in pivot_df.groupby(level=["direction", "fraction"]):
        row = {
            "direction": direction,
            "direction_label": DIRECTION_LABELS[direction],
            "fraction": float(fraction),
        }
        for metric in METRICS:
            rf_values = group[(metric, "rf")].to_numpy(dtype=float)
            mlp_values = group[(metric, "mlp")].to_numpy(dtype=float)
            deltas = mlp_values - rf_values
            stats = summarize_metric(pd.Series(deltas))
            positive, negative, nonzero, p_value = exact_sign_test_two_sided(deltas)
            for key, value in stats.items():
                row[f"{metric}_delta_{key}"] = value
            row[f"{metric}_delta_positive_seeds"] = positive
            row[f"{metric}_delta_negative_seeds"] = negative
            row[f"{metric}_delta_nonzero_seeds"] = nonzero
            row[f"{metric}_delta_sign_test_pvalue"] = p_value
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["direction", "fraction"]).reset_index(drop=True)


def build_paired_curve_deltas(curve_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "auc_f1_attack_normalized",
        "auc_recall_attack_normalized",
        "auc_f1_macro_normalized",
        "auc_accuracy_normalized",
        "low_fraction_mean_f1_attack",
        "low_fraction_mean_recall_attack",
        "low_fraction_mean_f1_macro",
        "low_fraction_mean_accuracy",
        "full_fraction_f1_attack",
        "full_fraction_recall_attack",
        "full_fraction_f1_macro",
        "full_fraction_accuracy",
    ]
    rows = []
    pivot_df = curve_df.pivot_table(
        index=["direction", "seed"],
        columns="model",
        values=metrics,
        aggfunc="first",
    )
    if pivot_df.empty:
        return pd.DataFrame()

    for direction, group in pivot_df.groupby(level="direction"):
        row = {
            "direction": direction,
            "direction_label": DIRECTION_LABELS[direction],
        }
        for metric in metrics:
            rf_values = group[(metric, "rf")].to_numpy(dtype=float)
            mlp_values = group[(metric, "mlp")].to_numpy(dtype=float)
            deltas = mlp_values - rf_values
            stats = summarize_metric(pd.Series(deltas))
            positive, negative, nonzero, p_value = exact_sign_test_two_sided(deltas)
            for key, value in stats.items():
                row[f"{metric}_delta_{key}"] = value
            row[f"{metric}_delta_positive_seeds"] = positive
            row[f"{metric}_delta_negative_seeds"] = negative
            row[f"{metric}_delta_nonzero_seeds"] = nonzero
            row[f"{metric}_delta_sign_test_pvalue"] = p_value
        rows.append(row)

    return pd.DataFrame(rows).sort_values("direction").reset_index(drop=True)


def plot_metric_curves_with_ci(summary_stats_df: pd.DataFrame, out_path: Path, metric: str, title: str) -> None:
    plt.figure(figsize=(10, 6))
    colors = {
        ("iot23_to_unsw", "rf"): "#F58518",
        ("iot23_to_unsw", "mlp"): "#4C78A8",
        ("unsw_to_iot23", "rf"): "#54A24B",
        ("unsw_to_iot23", "mlp"): "#E45756",
    }

    for direction in ["iot23_to_unsw", "unsw_to_iot23"]:
        for model in ["rf", "mlp"]:
            subset = summary_stats_df[
                (summary_stats_df["direction"] == direction) & (summary_stats_df["model"] == model)
            ].sort_values("fraction")
            if subset.empty:
                continue

            x = subset["fraction"].to_numpy(dtype=float)
            mean = subset[f"{metric}_mean"].to_numpy(dtype=float)
            low = subset[f"{metric}_ci95_low"].to_numpy(dtype=float)
            high = subset[f"{metric}_ci95_high"].to_numpy(dtype=float)
            label = f"{subset['direction_label'].iloc[0]} {subset['model_label'].iloc[0]}"
            color = colors[(direction, model)]

            plt.plot(x, mean, marker="o", linewidth=2, color=color, label=label)
            plt.fill_between(x, low, high, color=color, alpha=0.18)

    plt.title(title)
    plt.xlabel("Prefix Fraction")
    plt.ylabel(metric)
    plt.xticks(sorted(summary_stats_df["fraction"].dropna().unique().tolist()))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_delta_curves(delta_df: pd.DataFrame, out_path: Path, metric: str, title: str) -> None:
    plt.figure(figsize=(9, 5.5))
    plt.axhline(0.0, color="black", linewidth=1, alpha=0.5)

    for direction, color in [("iot23_to_unsw", "#4C78A8"), ("unsw_to_iot23", "#E45756")]:
        subset = delta_df[delta_df["direction"] == direction].sort_values("fraction")
        if subset.empty:
            continue

        x = subset["fraction"].to_numpy(dtype=float)
        mean = subset[f"{metric}_delta_mean"].to_numpy(dtype=float)
        low = subset[f"{metric}_delta_ci95_low"].to_numpy(dtype=float)
        high = subset[f"{metric}_delta_ci95_high"].to_numpy(dtype=float)
        label = f"{subset['direction_label'].iloc[0]} MLP - RF"

        plt.plot(x, mean, marker="o", linewidth=2, color=color, label=label)
        plt.fill_between(x, low, high, color=color, alpha=0.18)

    plt.title(title)
    plt.xlabel("Prefix Fraction")
    plt.ylabel(f"Delta {metric}")
    plt.xticks(sorted(delta_df["fraction"].dropna().unique().tolist()))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def validate_run_matching(summary_df: pd.DataFrame) -> pd.DataFrame:
    validation_rows = []
    for direction, direction_df in summary_df.groupby("direction", sort=True):
        model_summary = (
            direction_df.groupby("model", as_index=False)
            .agg(
                n_seeds=("seed", "nunique"),
                fraction_count=("fraction", "nunique"),
                min_fraction=("fraction", "min"),
                max_fraction=("fraction", "max"),
                source_train_rows_config=("source_train_rows_config", "first"),
                target_val_rows_config=("target_val_rows_config", "first"),
                target_test_rows_config=("target_test_rows_config", "first"),
                n_aligned_features_config=("n_aligned_features_config", "first"),
            )
            .sort_values("model")
            .reset_index(drop=True)
        )
        validation_rows.extend(model_summary.assign(direction=direction).to_dict(orient="records"))
    return pd.DataFrame(validation_rows)


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

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary_df, inventory_df = collect_run_data(runs_dir)
    for col in [
        "fraction",
        "source_train_rows_config",
        "target_val_rows_config",
        "target_test_rows_config",
        "seed",
        "config_seed",
        "n_aligned_features_config",
    ]:
        if col in summary_df.columns:
            summary_df[col] = pd.to_numeric(summary_df[col], errors="coerce")

    summary_stats_df = aggregate_per_fraction(summary_df)
    curve_seed_df = build_curve_level_seed_summary(summary_df)
    curve_stats_df = aggregate_curve_level(curve_seed_df)
    paired_fraction_df = build_paired_fraction_deltas(summary_df)
    paired_curve_df = build_paired_curve_deltas(curve_seed_df)
    validation_df = validate_run_matching(summary_df)

    inventory_df.to_csv(out_dir / "run_inventory.csv", index=False)
    summary_df.sort_values(["direction", "model", "seed", "fraction"]).to_csv(
        out_dir / "all_seed_test_rows.csv",
        index=False,
    )
    summary_stats_df.to_csv(out_dir / "per_fraction_summary_stats.csv", index=False)
    curve_seed_df.to_csv(out_dir / "curve_level_seed_summary.csv", index=False)
    curve_stats_df.to_csv(out_dir / "curve_level_summary_stats.csv", index=False)
    paired_fraction_df.to_csv(out_dir / "paired_fraction_deltas.csv", index=False)
    paired_curve_df.to_csv(out_dir / "paired_curve_deltas.csv", index=False)
    validation_df.to_csv(out_dir / "run_matching_validation.csv", index=False)

    plot_metric_curves_with_ci(
        summary_stats_df,
        plots_dir / "f1_attack_mean_ci_vs_fraction.png",
        metric="f1_attack",
        title="Multi-Seed Cross-Domain Test Attack F1 vs Prefix Fraction",
    )
    plot_metric_curves_with_ci(
        summary_stats_df,
        plots_dir / "recall_attack_mean_ci_vs_fraction.png",
        metric="recall_attack",
        title="Multi-Seed Cross-Domain Test Attack Recall vs Prefix Fraction",
    )
    plot_metric_curves_with_ci(
        summary_stats_df,
        plots_dir / "f1_macro_mean_ci_vs_fraction.png",
        metric="f1_macro",
        title="Multi-Seed Cross-Domain Test Macro F1 vs Prefix Fraction",
    )
    plot_metric_curves_with_ci(
        summary_stats_df,
        plots_dir / "accuracy_mean_ci_vs_fraction.png",
        metric="accuracy",
        title="Multi-Seed Cross-Domain Test Accuracy vs Prefix Fraction",
    )
    plot_delta_curves(
        paired_fraction_df,
        plots_dir / "f1_attack_delta_mlp_minus_rf.png",
        metric="f1_attack",
        title="Multi-Seed Delta: MLP - RF Attack F1",
    )
    plot_delta_curves(
        paired_fraction_df,
        plots_dir / "recall_attack_delta_mlp_minus_rf.png",
        metric="recall_attack",
        title="Multi-Seed Delta: MLP - RF Attack Recall",
    )

    save_json(
        {
            "runs_dir": str(runs_dir),
            "out_dir": str(out_dir),
            "n_total_runs": int(inventory_df.shape[0]),
            "n_unique_seeds": int(inventory_df["seed"].nunique()),
            "directions": sorted(inventory_df["direction"].unique().tolist()),
            "models": sorted(inventory_df["model"].unique().tolist()),
        },
        out_dir / "analysis_manifest.json",
    )


if __name__ == "__main__":
    main()
