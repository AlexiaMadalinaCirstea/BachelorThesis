from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASELINE_RUN_PATTERN = re.compile(r"^outputs_(iot23_to_unsw|unsw_to_iot23)_(rf|mlp)_seed(\d+)$")
SENSITIVITY_RUN_PATTERN = re.compile(r"^outputs_(iot23_to_unsw|unsw_to_iot23)_(rf|mlp)_(size|eval_cap)_seed(\d+)$")
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
STUDY_LABELS = {
    "size": "Size sensitivity",
    "eval_cap": "Eval-cap sensitivity",
}


def trapezoid_auc(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze cross-domain size/eval-cap sensitivity experiments against the repeated baseline."
    )
    parser.add_argument(
        "--baseline_dir",
        default="early_detection/cross_domain_early_detection/multiple_seeds_test",
        help="Directory containing baseline repeated-seed runs.",
    )
    parser.add_argument(
        "--sensitivity_dir",
        default="early_detection/cross_domain_early_detection/sensitivity_tests",
        help="Directory containing sensitivity-study runs.",
    )
    parser.add_argument(
        "--out_dir",
        default="early_detection/cross_domain_early_detection/sensitivity_analyzer",
        help="Directory for aggregated sensitivity outputs.",
    )
    return parser.parse_args()


def save_json(obj: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True)


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


def extract_source_train_rows(run_config: dict) -> int | None:
    return run_config.get("iot_train_rows") or run_config.get("unsw_source_train_rows")


def extract_target_val_rows(run_config: dict) -> int | None:
    return run_config.get("unsw_target_val_rows") or run_config.get("iot_val_rows")


def extract_target_test_rows(run_config: dict) -> int | None:
    return run_config.get("unsw_test_rows") or run_config.get("iot_test_rows")


def parse_baseline_run_name(name: str) -> dict | None:
    match = BASELINE_RUN_PATTERN.match(name)
    if not match:
        return None
    direction, model, seed = match.groups()
    return {
        "direction": direction,
        "model": model,
        "seed": int(seed),
        "run_name": name,
    }


def parse_sensitivity_run_name(name: str) -> dict | None:
    match = SENSITIVITY_RUN_PATTERN.match(name)
    if not match:
        return None
    direction, model, study, seed = match.groups()
    return {
        "direction": direction,
        "model": model,
        "study": study,
        "seed": int(seed),
        "run_name": name,
    }


def read_run_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def sensitivity_config_label(study: str, direction: str, run_config: dict) -> str:
    if study == "size":
        if direction == "iot23_to_unsw":
            return f"IoT train {int(run_config.get('iot_train_rows', 0))}"
        return f"UNSW train {int(run_config.get('unsw_source_train_rows', 0))}"
    if direction == "iot23_to_unsw":
        return f"UNSW eval {int(run_config.get('unsw_target_val_rows', 0))}"
    return f"IoT eval {int(run_config.get('iot_val_rows', 0))}"


def load_overall_fraction_summary(
    run_dir: Path,
    run_meta: dict,
    run_config: dict,
    study: str | None,
    config_group: str,
    config_label: str,
) -> pd.DataFrame:
    summary_path = run_dir / "overall_fraction_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")

    df = pd.read_csv(summary_path)
    df = df[df["split"] == "test"].copy()
    if df.empty:
        raise AssertionError(f"No test rows found in {summary_path}")

    for key, value in run_meta.items():
        df[key] = value

    df["study"] = study
    df["study_label"] = STUDY_LABELS.get(study, "")
    df["config_group"] = config_group
    df["config_label"] = config_label
    df["direction_label"] = df["direction"].map(DIRECTION_LABELS)
    df["model_label"] = df["model"].map(MODEL_LABELS)
    df["source_train_rows_config"] = extract_source_train_rows(run_config)
    df["target_val_rows_config"] = extract_target_val_rows(run_config)
    df["target_test_rows_config"] = extract_target_test_rows(run_config)
    df["config_seed"] = run_config.get("seed")
    return df


def collect_baseline_data(runs_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_rows = []
    inventory_rows = []

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        run_meta = parse_baseline_run_name(run_dir.name)
        if run_meta is None:
            continue

        config_path = run_dir / "run_config.json"
        if not config_path.exists():
            continue
        run_config = read_run_config(config_path)
        summary_df = load_overall_fraction_summary(run_dir, run_meta, run_config, None, "baseline", "baseline")
        all_rows.append(summary_df)
        inventory_rows.append(
            {
                **run_meta,
                "source_train_rows_config": extract_source_train_rows(run_config),
                "target_val_rows_config": extract_target_val_rows(run_config),
                "target_test_rows_config": extract_target_test_rows(run_config),
                "fraction_count": int(summary_df["fraction"].nunique()),
            }
        )

    if not all_rows:
        raise AssertionError(f"No baseline runs found in {runs_dir}")

    combined_df = pd.concat(all_rows, ignore_index=True)
    inventory_df = pd.DataFrame(inventory_rows).sort_values(["direction", "model", "seed"]).reset_index(drop=True)
    return combined_df, inventory_df


def collect_sensitivity_data(runs_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_rows = []
    inventory_rows = []

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        run_meta = parse_sensitivity_run_name(run_dir.name)
        if run_meta is None:
            continue

        config_path = run_dir / "run_config.json"
        if not config_path.exists():
            continue
        run_config = read_run_config(config_path)
        label = sensitivity_config_label(run_meta["study"], run_meta["direction"], run_config)
        summary_df = load_overall_fraction_summary(
            run_dir,
            run_meta,
            run_config,
            run_meta["study"],
            "sensitivity",
            label,
        )
        all_rows.append(summary_df)
        inventory_rows.append(
            {
                **run_meta,
                "config_label": label,
                "source_train_rows_config": extract_source_train_rows(run_config),
                "target_val_rows_config": extract_target_val_rows(run_config),
                "target_test_rows_config": extract_target_test_rows(run_config),
                "fraction_count": int(summary_df["fraction"].nunique()),
            }
        )

    if not all_rows:
        raise AssertionError(f"No sensitivity runs found in {runs_dir}")

    combined_df = pd.concat(all_rows, ignore_index=True)
    inventory_df = pd.DataFrame(inventory_rows).sort_values(["study", "direction", "model", "seed"]).reset_index(drop=True)
    return combined_df, inventory_df


def build_comparison_dataset(baseline_df: pd.DataFrame, sensitivity_df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for (study, direction, model), sens_group in sensitivity_df.groupby(["study", "direction", "model"], sort=True):
        seeds = sorted(sens_group["seed"].unique().tolist())
        base_group = baseline_df[
            (baseline_df["direction"] == direction)
            & (baseline_df["model"] == model)
            & (baseline_df["seed"].isin(seeds))
        ].copy()
        if base_group.empty:
            raise AssertionError(f"Missing baseline runs for study={study}, direction={direction}, model={model}")

        fraction_set = sorted(sens_group["fraction"].unique().tolist())
        baseline_fraction_set = sorted(base_group["fraction"].unique().tolist())
        if fraction_set != baseline_fraction_set:
            raise AssertionError(
                f"Fraction mismatch between baseline and sensitivity for study={study}, direction={direction}, model={model}"
            )

        base_group["study"] = study
        base_group["study_label"] = STUDY_LABELS[study]
        base_group["config_group"] = "baseline"
        base_group["config_label"] = "baseline"
        parts.append(base_group)
        parts.append(sens_group.copy())

    combined_df = pd.concat(parts, ignore_index=True)
    return combined_df.sort_values(["study", "direction", "model", "config_group", "seed", "fraction"]).reset_index(drop=True)


def aggregate_per_fraction(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = summary_df.groupby(["study", "config_group", "config_label", "direction", "model", "fraction"], sort=True)

    for (study, config_group, config_label, direction, model, fraction), group in grouped:
        row = {
            "study": study,
            "study_label": group["study_label"].iloc[0],
            "config_group": config_group,
            "config_label": config_label,
            "direction": direction,
            "direction_label": group["direction_label"].iloc[0],
            "model": model,
            "model_label": group["model_label"].iloc[0],
            "fraction": float(fraction),
            "source_train_rows_config": int(group["source_train_rows_config"].iloc[0]),
            "target_val_rows_config": int(group["target_val_rows_config"].iloc[0]),
            "target_test_rows_config": int(group["target_test_rows_config"].iloc[0]),
        }
        for metric in METRICS:
            stats = summarize_metric(group[metric])
            for key, value in stats.items():
                row[f"{metric}_{key}"] = value
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["study", "direction", "model", "config_group", "fraction"]).reset_index(drop=True)


def build_curve_level_seed_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["study", "config_group", "config_label", "direction", "model", "seed"]
    for (study, config_group, config_label, direction, model, seed), group in summary_df.groupby(group_cols, sort=True):
        ordered = group.sort_values("fraction")
        fractions = ordered["fraction"].to_numpy(dtype=float)
        span = float(fractions[-1] - fractions[0]) if len(fractions) > 1 else 0.0

        def normalized_auc(metric: str) -> float:
            values = ordered[metric].to_numpy(dtype=float)
            auc_value = trapezoid_auc(values, fractions)
            return float(auc_value / span) if span > 0 else float(values[0])

        low_fraction_df = ordered[ordered["fraction"] <= LOW_FRACTION_MAX]
        full_fraction_row = ordered.loc[ordered["fraction"] == ordered["fraction"].max()].iloc[0]
        rows.append(
            {
                "study": study,
                "study_label": ordered["study_label"].iloc[0],
                "config_group": config_group,
                "config_label": config_label,
                "direction": direction,
                "direction_label": ordered["direction_label"].iloc[0],
                "model": model,
                "model_label": ordered["model_label"].iloc[0],
                "seed": int(seed),
                "auc_f1_attack_normalized": normalized_auc("f1_attack"),
                "auc_recall_attack_normalized": normalized_auc("recall_attack"),
                "auc_f1_macro_normalized": normalized_auc("f1_macro"),
                "auc_accuracy_normalized": normalized_auc("accuracy"),
                "low_fraction_mean_f1_attack": float(low_fraction_df["f1_attack"].mean()),
                "low_fraction_mean_recall_attack": float(low_fraction_df["recall_attack"].mean()),
                "full_fraction_f1_attack": float(full_fraction_row["f1_attack"]),
                "full_fraction_recall_attack": float(full_fraction_row["recall_attack"]),
            }
        )

    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def aggregate_curve_level(curve_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = [
        "auc_f1_attack_normalized",
        "auc_recall_attack_normalized",
        "auc_f1_macro_normalized",
        "auc_accuracy_normalized",
        "low_fraction_mean_f1_attack",
        "low_fraction_mean_recall_attack",
        "full_fraction_f1_attack",
        "full_fraction_recall_attack",
    ]

    group_cols = ["study", "config_group", "config_label", "direction", "model"]
    for (study, config_group, config_label, direction, model), group in curve_df.groupby(group_cols, sort=True):
        row = {
            "study": study,
            "study_label": group["study_label"].iloc[0],
            "config_group": config_group,
            "config_label": config_label,
            "direction": direction,
            "direction_label": group["direction_label"].iloc[0],
            "model": model,
            "model_label": group["model_label"].iloc[0],
        }
        for metric in metrics:
            stats = summarize_metric(group[metric])
            for key, value in stats.items():
                row[f"{metric}_{key}"] = value
        rows.append(row)

    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def build_paired_fraction_deltas(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pivot_df = summary_df.pivot_table(
        index=["study", "direction", "model", "seed", "fraction"],
        columns="config_group",
        values=METRICS,
        aggfunc="first",
    )
    if pivot_df.empty:
        return pd.DataFrame()
    available_configs = set(pivot_df.columns.get_level_values(1))
    if not {"baseline", "sensitivity"}.issubset(available_configs):
        raise AssertionError("Sensitivity delta analysis requires both baseline and sensitivity rows.")

    for (study, direction, model, fraction), group in pivot_df.groupby(level=["study", "direction", "model", "fraction"]):
        row = {
            "study": study,
            "study_label": STUDY_LABELS[study],
            "direction": direction,
            "direction_label": DIRECTION_LABELS[direction],
            "model": model,
            "model_label": MODEL_LABELS[model],
            "fraction": float(fraction),
        }
        for metric in METRICS:
            baseline_values = group[(metric, "baseline")].to_numpy(dtype=float)
            sensitivity_values = group[(metric, "sensitivity")].to_numpy(dtype=float)
            deltas = sensitivity_values - baseline_values
            stats = summarize_metric(pd.Series(deltas))
            positive, negative, nonzero, p_value = exact_sign_test_two_sided(deltas)
            for key, value in stats.items():
                row[f"{metric}_delta_{key}"] = value
            row[f"{metric}_delta_positive_seeds"] = positive
            row[f"{metric}_delta_negative_seeds"] = negative
            row[f"{metric}_delta_nonzero_seeds"] = nonzero
            row[f"{metric}_delta_sign_test_pvalue"] = p_value
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["study", "direction", "model", "fraction"]).reset_index(drop=True)


def build_paired_curve_deltas(curve_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = [
        "auc_f1_attack_normalized",
        "auc_recall_attack_normalized",
        "auc_f1_macro_normalized",
        "auc_accuracy_normalized",
        "low_fraction_mean_f1_attack",
        "low_fraction_mean_recall_attack",
        "full_fraction_f1_attack",
        "full_fraction_recall_attack",
    ]
    pivot_df = curve_df.pivot_table(
        index=["study", "direction", "model", "seed"],
        columns="config_group",
        values=metrics,
        aggfunc="first",
    )
    if pivot_df.empty:
        return pd.DataFrame()

    for (study, direction, model), group in pivot_df.groupby(level=["study", "direction", "model"]):
        row = {
            "study": study,
            "study_label": STUDY_LABELS[study],
            "direction": direction,
            "direction_label": DIRECTION_LABELS[direction],
            "model": model,
            "model_label": MODEL_LABELS[model],
        }
        for metric in metrics:
            baseline_values = group[(metric, "baseline")].to_numpy(dtype=float)
            sensitivity_values = group[(metric, "sensitivity")].to_numpy(dtype=float)
            deltas = sensitivity_values - baseline_values
            stats = summarize_metric(pd.Series(deltas))
            positive, negative, nonzero, p_value = exact_sign_test_two_sided(deltas)
            for key, value in stats.items():
                row[f"{metric}_delta_{key}"] = value
            row[f"{metric}_delta_positive_seeds"] = positive
            row[f"{metric}_delta_negative_seeds"] = negative
            row[f"{metric}_delta_nonzero_seeds"] = nonzero
            row[f"{metric}_delta_sign_test_pvalue"] = p_value
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["study", "direction", "model"]).reset_index(drop=True)


def plot_study_metric_curves(
    summary_stats_df: pd.DataFrame,
    study: str,
    direction: str,
    metric: str,
    out_path: Path,
) -> None:
    subset = summary_stats_df[
        (summary_stats_df["study"] == study)
        & (summary_stats_df["direction"] == direction)
    ].copy()
    if subset.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"rf": "#F58518", "mlp": "#4C78A8"}
    linestyles = {"baseline": "--", "sensitivity": "-"}
    alphas = {"baseline": 0.12, "sensitivity": 0.18}

    for model in ["rf", "mlp"]:
        for config_group in ["baseline", "sensitivity"]:
            group = subset[
                (subset["model"] == model) & (subset["config_group"] == config_group)
            ].sort_values("fraction")
            if group.empty:
                continue
            x = group["fraction"].to_numpy(dtype=float)
            mean = group[f"{metric}_mean"].to_numpy(dtype=float)
            low = group[f"{metric}_ci95_low"].to_numpy(dtype=float)
            high = group[f"{metric}_ci95_high"].to_numpy(dtype=float)
            label_suffix = group["config_label"].iloc[0]
            label = f"{MODEL_LABELS[model]} {label_suffix}"
            ax.plot(
                x,
                mean,
                marker="o",
                linewidth=2,
                color=colors[model],
                linestyle=linestyles[config_group],
                label=label,
            )
            ax.fill_between(x, low, high, color=colors[model], alpha=alphas[config_group])

    ax.set_title(f"{STUDY_LABELS[study]}: {DIRECTION_LABELS[direction]} {metric}")
    ax.set_xlabel("Prefix Fraction")
    ax.set_ylabel(metric)
    ax.set_xticks(sorted(subset["fraction"].dropna().unique().tolist()))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_study_delta_curves(
    delta_df: pd.DataFrame,
    study: str,
    direction: str,
    metric: str,
    out_path: Path,
) -> None:
    subset = delta_df[
        (delta_df["study"] == study)
        & (delta_df["direction"] == direction)
    ].copy()
    if subset.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.5)
    colors = {"rf": "#F58518", "mlp": "#4C78A8"}

    for model in ["rf", "mlp"]:
        group = subset[subset["model"] == model].sort_values("fraction")
        if group.empty:
            continue
        x = group["fraction"].to_numpy(dtype=float)
        mean = group[f"{metric}_delta_mean"].to_numpy(dtype=float)
        low = group[f"{metric}_delta_ci95_low"].to_numpy(dtype=float)
        high = group[f"{metric}_delta_ci95_high"].to_numpy(dtype=float)
        ax.plot(x, mean, marker="o", linewidth=2, color=colors[model], label=f"{MODEL_LABELS[model]} sensitivity - baseline")
        ax.fill_between(x, low, high, color=colors[model], alpha=0.18)

    ax.set_title(f"{STUDY_LABELS[study]}: {DIRECTION_LABELS[direction]} delta {metric}")
    ax.set_xlabel("Prefix Fraction")
    ax.set_ylabel(f"Delta {metric}")
    ax.set_xticks(sorted(subset["fraction"].dropna().unique().tolist()))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def validate_matching(comparison_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (study, direction, model), group in comparison_df.groupby(["study", "direction", "model"], sort=True):
        summary = (
            group.groupby(["config_group", "config_label"], as_index=False)
            .agg(
                n_seeds=("seed", "nunique"),
                fraction_count=("fraction", "nunique"),
                min_fraction=("fraction", "min"),
                max_fraction=("fraction", "max"),
                source_train_rows_config=("source_train_rows_config", "first"),
                target_val_rows_config=("target_val_rows_config", "first"),
                target_test_rows_config=("target_test_rows_config", "first"),
            )
            .sort_values(["config_group", "config_label"])
            .reset_index(drop=True)
        )
        summary["study"] = study
        summary["direction"] = direction
        summary["model"] = model
        rows.extend(summary.to_dict(orient="records"))
    return pd.DataFrame(rows).sort_values(["study", "direction", "model", "config_group", "config_label"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    thesis_root = script_dir.parents[1]

    baseline_dir = Path(args.baseline_dir)
    if not baseline_dir.is_absolute():
        baseline_dir = (thesis_root / baseline_dir).resolve()

    sensitivity_dir = Path(args.sensitivity_dir)
    if not sensitivity_dir.is_absolute():
        sensitivity_dir = (thesis_root / sensitivity_dir).resolve()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (thesis_root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    baseline_df, baseline_inventory_df = collect_baseline_data(baseline_dir)
    sensitivity_df, sensitivity_inventory_df = collect_sensitivity_data(sensitivity_dir)
    comparison_df = build_comparison_dataset(baseline_df, sensitivity_df)

    for col in [
        "fraction",
        "source_train_rows_config",
        "target_val_rows_config",
        "target_test_rows_config",
        "seed",
        "config_seed",
    ]:
        if col in comparison_df.columns:
            comparison_df[col] = pd.to_numeric(comparison_df[col], errors="coerce")

    per_fraction_df = aggregate_per_fraction(comparison_df)
    curve_seed_df = build_curve_level_seed_summary(comparison_df)
    curve_stats_df = aggregate_curve_level(curve_seed_df)
    paired_fraction_df = build_paired_fraction_deltas(comparison_df)
    paired_curve_df = build_paired_curve_deltas(curve_seed_df)
    matching_df = validate_matching(comparison_df)

    baseline_inventory_df.to_csv(out_dir / "baseline_run_inventory.csv", index=False)
    sensitivity_inventory_df.to_csv(out_dir / "sensitivity_run_inventory.csv", index=False)
    comparison_df.to_csv(out_dir / "comparison_test_rows.csv", index=False)
    per_fraction_df.to_csv(out_dir / "per_fraction_summary_stats.csv", index=False)
    curve_seed_df.to_csv(out_dir / "curve_level_seed_summary.csv", index=False)
    curve_stats_df.to_csv(out_dir / "curve_level_summary_stats.csv", index=False)
    paired_fraction_df.to_csv(out_dir / "paired_fraction_deltas.csv", index=False)
    paired_curve_df.to_csv(out_dir / "paired_curve_deltas.csv", index=False)
    matching_df.to_csv(out_dir / "matching_validation.csv", index=False)

    for study in sorted(comparison_df["study"].dropna().unique().tolist()):
        for direction in sorted(comparison_df["direction"].dropna().unique().tolist()):
            plot_study_metric_curves(
                per_fraction_df,
                study,
                direction,
                "f1_attack",
                plots_dir / f"{study}_{direction}_f1_attack.png",
            )
            plot_study_metric_curves(
                per_fraction_df,
                study,
                direction,
                "recall_attack",
                plots_dir / f"{study}_{direction}_recall_attack.png",
            )
            plot_study_delta_curves(
                paired_fraction_df,
                study,
                direction,
                "f1_attack",
                plots_dir / f"{study}_{direction}_f1_attack_delta.png",
            )
            plot_study_delta_curves(
                paired_fraction_df,
                study,
                direction,
                "recall_attack",
                plots_dir / f"{study}_{direction}_recall_attack_delta.png",
            )

    save_json(
        {
            "baseline_dir": str(baseline_dir),
            "sensitivity_dir": str(sensitivity_dir),
            "out_dir": str(out_dir),
            "n_baseline_runs_used": int(baseline_inventory_df.shape[0]),
            "n_sensitivity_runs": int(sensitivity_inventory_df.shape[0]),
            "studies": sorted(sensitivity_inventory_df["study"].unique().tolist()),
        },
        out_dir / "analysis_manifest.json",
    )


if __name__ == "__main__":
    main()
