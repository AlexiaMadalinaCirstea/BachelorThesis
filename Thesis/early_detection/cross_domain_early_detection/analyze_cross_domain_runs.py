from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def trapezoid_auc(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze RF/MLP cross-domain early-detection runs and generate matched comparison artifacts."
    )
    parser.add_argument(
        "--runs_dir",
        default="early_detection/cross_domain_early_detection",
        help="Directory containing run folders.",
    )
    parser.add_argument(
        "--out_dir",
        default="early_detection/cross_domain_early_detection/fixed_analyzer",
        help="Directory for merged CSVs and plots.",
    )
    return parser.parse_args()


def save_json(obj: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True)


def parse_run_folder_name(name: str) -> dict[str, object]:
    exp_match = re.search(r"exp(\d+)", name)
    seed_match = re.search(r"seed[_-]?(\d+)", name)
    return {
        "run_name": name,
        "exp_number": int(exp_match.group(1)) if exp_match else 0,
        "seed_number": int(seed_match.group(1)) if seed_match else None,
    }


def read_run_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_overall_fraction_summary(run_dir: Path, run_meta: dict, run_config: dict) -> pd.DataFrame:
    summary_path = run_dir / "overall_fraction_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing overall_fraction_summary.csv in {run_dir}")

    df = pd.read_csv(summary_path)
    df["run_name"] = run_meta["run_name"]
    df["exp_number"] = run_meta["exp_number"]
    df["seed_number"] = run_meta["seed_number"]
    df["model"] = run_config.get("model", "")
    df["include_review_features"] = bool(run_config.get("include_review_features", False))
    df["n_aligned_features_config"] = run_config.get("n_aligned_features")
    df["config_seed"] = run_config.get("seed")
    df["rf_n_estimators"] = run_config.get("rf_n_estimators")
    df["rf_max_depth"] = run_config.get("rf_max_depth")
    df["mlp_hidden_layers"] = str(run_config.get("mlp_hidden_layers")) if run_config.get("model") == "mlp" else ""
    df["mlp_max_iter"] = run_config.get("mlp_max_iter") if run_config.get("model") == "mlp" else None
    df["mlp_batch_size"] = run_config.get("mlp_batch_size") if run_config.get("model") == "mlp" else None
    return df


def collect_run_data(runs_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_parts = []
    run_rows = []

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        if run_dir.name.startswith("smoke_") or run_dir.name.startswith("__"):
            continue
        config_path = run_dir / "run_config.json"
        if not config_path.exists():
            continue

        run_meta = parse_run_folder_name(run_dir.name)
        run_config = read_run_config(config_path)
        summary_df = load_overall_fraction_summary(run_dir, run_meta, run_config)
        summary_parts.append(summary_df)
        run_rows.append(
            {
                "run_name": run_meta["run_name"],
                "exp_number": run_meta["exp_number"],
                "model": run_config.get("model"),
                "include_review_features": bool(run_config.get("include_review_features", False)),
                "n_aligned_features": run_config.get("n_aligned_features"),
                "rf_n_estimators": run_config.get("rf_n_estimators"),
                "rf_max_depth": run_config.get("rf_max_depth"),
                "mlp_hidden_layers": str(run_config.get("mlp_hidden_layers")) if run_config.get("model") == "mlp" else "",
                "mlp_max_iter": run_config.get("mlp_max_iter") if run_config.get("model") == "mlp" else None,
                "mlp_batch_size": run_config.get("mlp_batch_size") if run_config.get("model") == "mlp" else None,
                "seed": run_config.get("seed"),
            }
        )

    if not summary_parts:
        raise AssertionError(f"No run folders with run_config.json found in {runs_dir}")

    summary_all_df = pd.concat(summary_parts, ignore_index=True)
    run_manifest_df = pd.DataFrame(run_rows).sort_values(["model", "run_name"]).reset_index(drop=True)
    return summary_all_df, run_manifest_df


def build_fraction_grid_table(summary_all_df: pd.DataFrame) -> pd.DataFrame:
    test_df = summary_all_df[summary_all_df["split"] == "test"].copy()
    val_df = summary_all_df[summary_all_df["split"] == "val"].copy()
    if test_df.empty:
        raise AssertionError("No test rows available to build the comparison signature table.")

    test_grouped = (
        test_df.groupby(["direction", "model", "run_name", "exp_number"], as_index=False)
        .agg(
            fraction_count=("fraction", "nunique"),
            max_fraction=("fraction", "max"),
            min_fraction=("fraction", "min"),
            signature_test_rows_config=("rows_evaluated", "max"),
        )
    )

    val_grouped = (
        val_df.groupby(["direction", "model", "run_name", "exp_number"], as_index=False)
        .agg(signature_val_rows_config=("rows_evaluated", "max"))
    )

    return test_grouped.merge(
        val_grouped,
        on=["direction", "model", "run_name", "exp_number"],
        how="left",
    )


def select_best_runs(summary_all_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_full = summary_all_df[
        (summary_all_df["split"] == "test") & (summary_all_df["fraction"] == summary_all_df["fraction"].max())
    ].copy()
    if test_full.empty:
        raise AssertionError("No full-fraction test rows found for best-run selection.")

    fraction_grid_df = build_fraction_grid_table(summary_all_df)
    test_full = test_full.merge(
        fraction_grid_df,
        on=["direction", "model", "run_name", "exp_number"],
        how="left",
    )

    selected_rows: list[pd.Series] = []
    selection_rows: list[dict[str, object]] = []

    for direction, direction_df in test_full.groupby("direction", sort=True):
        models = sorted(direction_df["model"].dropna().unique().tolist())
        required_model_count = len(models)
        if required_model_count == 0:
            continue

        signature_df = (
            direction_df.groupby(
                [
                    "fraction_count",
                    "max_fraction",
                    "min_fraction",
                    "signature_val_rows_config",
                    "signature_test_rows_config",
                ],
                as_index=False,
            )["model"]
            .nunique()
            .rename(columns={"model": "n_models"})
        )
        valid_signatures = signature_df[signature_df["n_models"] == required_model_count].copy()
        if valid_signatures.empty:
            raise AssertionError(f"No matched comparison signature found across models for direction={direction}.")

        chosen_signature = valid_signatures.sort_values(
            [
                "fraction_count",
                "max_fraction",
                "signature_test_rows_config",
                "signature_val_rows_config",
                "min_fraction",
            ],
            ascending=[False, False, False, False, True],
        ).iloc[0]

        signature_mask = (
            (direction_df["fraction_count"] == chosen_signature["fraction_count"])
            & (direction_df["max_fraction"] == chosen_signature["max_fraction"])
            & (direction_df["min_fraction"] == chosen_signature["min_fraction"])
            & (direction_df["signature_val_rows_config"] == chosen_signature["signature_val_rows_config"])
            & (direction_df["signature_test_rows_config"] == chosen_signature["signature_test_rows_config"])
        )
        matched_df = direction_df[signature_mask].copy()

        for model, model_df in matched_df.groupby("model", sort=True):
            chosen_run = model_df.sort_values(["f1_attack", "exp_number"], ascending=[False, False]).iloc[0]
            selected_rows.append(chosen_run)
            selection_rows.append(
                {
                    "direction": direction,
                    "model": model,
                    "selected_run_name": chosen_run["run_name"],
                    "selected_exp_number": int(chosen_run["exp_number"]),
                    "selected_f1_attack_full_fraction": float(chosen_run["f1_attack"]),
                    "fraction_count": int(chosen_signature["fraction_count"]),
                    "min_fraction": float(chosen_signature["min_fraction"]),
                    "max_fraction": float(chosen_signature["max_fraction"]),
                    "val_rows_config": int(chosen_signature["signature_val_rows_config"]),
                    "test_rows_config": int(chosen_signature["signature_test_rows_config"]),
                }
            )

    if not selected_rows:
        raise AssertionError("Best-run selection produced no rows.")

    selected_keys = {
        (row["direction"], row["model"], row["run_name"])
        for _, row in pd.DataFrame(selected_rows).iterrows()
    }
    selected_df = summary_all_df[
        summary_all_df.apply(
            lambda row: (row["direction"], row["model"], row["run_name"]) in selected_keys,
            axis=1,
        )
    ].copy()
    selection_manifest_df = pd.DataFrame(selection_rows).sort_values(["direction", "model"]).reset_index(drop=True)
    return selected_df, selection_manifest_df


def plot_metric_by_fraction(df: pd.DataFrame, direction: str, out_path: Path, metric: str, title: str) -> None:
    subset = df[(df["direction"] == direction) & (df["split"] == "test")].copy()
    if subset.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    for model, model_df in subset.groupby("model", sort=True):
        ordered = model_df.sort_values("fraction")
        ax.plot(ordered["fraction"], ordered[metric], marker="o", linewidth=2, label=model.upper())

    ax.set_xlabel("Prefix fraction")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def build_curve_level_summary(best_df: pd.DataFrame) -> pd.DataFrame:
    test_df = best_df[best_df["split"] == "test"].copy()
    rows = []

    for (direction, model, run_name), group in test_df.groupby(["direction", "model", "run_name"], sort=True):
        ordered = group.sort_values("fraction")
        x = ordered["fraction"].to_numpy(dtype=float)
        x_width = float(x.max() - x.min()) if len(x) > 1 else 0.0

        def normalized_auc(metric: str) -> float:
            y = ordered[metric].to_numpy(dtype=float)
            if len(y) == 1 or x_width == 0.0:
                return float(y[0])
            return float(trapezoid_auc(y, x) / x_width)

        low_fraction_mask = ordered["fraction"] <= 0.20
        low_fraction_df = ordered[low_fraction_mask]
        rows.append(
            {
                "direction": direction,
                "model": model,
                "run_name": run_name,
                "normalized_auc_f1_attack": normalized_auc("f1_attack"),
                "normalized_auc_recall_attack": normalized_auc("recall_attack"),
                "normalized_auc_f1_macro": normalized_auc("f1_macro"),
                "low_fraction_mean_f1_attack": float(low_fraction_df["f1_attack"].mean()),
                "low_fraction_mean_recall_attack": float(low_fraction_df["recall_attack"].mean()),
                "full_fraction_f1_attack": float(ordered.iloc[-1]["f1_attack"]),
                "full_fraction_recall_attack": float(ordered.iloc[-1]["recall_attack"]),
                "full_fraction_accuracy": float(ordered.iloc[-1]["accuracy"]),
            }
        )

    return pd.DataFrame(rows).sort_values(["direction", "model"]).reset_index(drop=True)


def build_low_fraction_summary(best_df: pd.DataFrame) -> pd.DataFrame:
    test_df = best_df[(best_df["split"] == "test") & (best_df["fraction"] <= 0.20)].copy()
    rows = []
    for (direction, model), group in test_df.groupby(["direction", "model"], sort=True):
        rows.append(
            {
                "direction": direction,
                "model": model,
                "mean_f1_attack": float(group["f1_attack"].mean()),
                "mean_recall_attack": float(group["recall_attack"].mean()),
                "mean_f1_macro": float(group["f1_macro"].mean()),
                "mean_accuracy": float(group["accuracy"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["direction", "model"]).reset_index(drop=True)


def load_selected_detail_table(
    runs_dir: Path,
    direction: str,
    run_name: str,
    split: str,
    filename: str,
) -> pd.DataFrame:
    path = runs_dir / run_name / direction / split / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing detail file: {path}")
    return pd.read_csv(path)


def plot_matrix_heatmap(
    matrix_df: pd.DataFrame,
    out_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    rows, cols = matrix_df.shape
    fig_width = max(9, cols * 1.2)
    fig_height = max(6, rows * 0.35)
    has_missing = bool(matrix_df.isna().to_numpy().any())
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#d9d9d9")
    masked_values = np.ma.masked_invalid(matrix_df.to_numpy(dtype=float))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(masked_values, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_xticks(range(cols))
    ax.set_xticklabels([str(col) for col in matrix_df.columns], rotation=45, ha="right")
    ax.set_yticks(range(rows))
    ax.set_yticklabels(matrix_df.index.tolist())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    if has_missing:
        fig.text(
            0.99,
            0.01,
            "Gray cells = N/A (category/scenario absent from that prefix)",
            ha="right",
            va="bottom",
            fontsize=9,
            color="dimgray",
        )
    fig.tight_layout(rect=(0.0, 0.03 if has_missing else 0.0, 1.0, 1.0))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def build_iot23_scenario_artifacts(
    selection_manifest_df: pd.DataFrame,
    runs_dir: Path,
    out_dir: Path,
) -> pd.DataFrame:
    rows = []
    for _, row in selection_manifest_df.iterrows():
        if row["direction"] != "unsw_to_iot23":
            continue

        detail_df = load_selected_detail_table(
            runs_dir=runs_dir,
            direction=row["direction"],
            run_name=row["selected_run_name"],
            split="test",
            filename="scenario_metrics_all_fractions.csv",
        )
        detail_df["model"] = row["model"]
        detail_df["run_name"] = row["selected_run_name"]
        rows.append(detail_df)

        matrix = detail_df.pivot_table(
            index="scenario",
            columns="fraction",
            values="f1_attack",
            aggfunc="first",
        ).sort_index()
        plot_matrix_heatmap(
            matrix_df=matrix,
            out_path=out_dir / f"{row['model']}_iot23_scenario_heatmap_f1_attack.png",
            title=f"IoT-23 scenario attack F1 heatmap ({row['model'].upper()})",
            xlabel="Fraction",
            ylabel="Scenario",
        )

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def build_unsw_attack_category_artifacts(
    selection_manifest_df: pd.DataFrame,
    runs_dir: Path,
    out_dir: Path,
) -> pd.DataFrame:
    rows = []
    for _, row in selection_manifest_df.iterrows():
        if row["direction"] != "iot23_to_unsw":
            continue

        detail_df = load_selected_detail_table(
            runs_dir=runs_dir,
            direction=row["direction"],
            run_name=row["selected_run_name"],
            split="test",
            filename="attack_cat_metrics_all_fractions.csv",
        )
        detail_df["model"] = row["model"]
        detail_df["run_name"] = row["selected_run_name"]
        rows.append(detail_df)

        matrix = detail_df.pivot_table(
            index="attack_cat",
            columns="fraction",
            values="recall_attack",
            aggfunc="first",
        ).sort_index()
        plot_matrix_heatmap(
            matrix_df=matrix,
            out_path=out_dir / f"{row['model']}_unsw_attack_category_heatmap_recall_attack.png",
            title=f"UNSW attack-category recall heatmap ({row['model'].upper()})",
            xlabel="Fraction",
            ylabel="Attack category",
        )

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    args = parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary_all_df, run_manifest_df = collect_run_data(runs_dir)
    summary_all_df.to_csv(out_dir / "all_run_fraction_summaries.csv", index=False)
    run_manifest_df.to_csv(out_dir / "run_manifest.csv", index=False)

    best_runs_df, selection_manifest_df = select_best_runs(summary_all_df)
    best_runs_df.to_csv(out_dir / "selected_best_run_fraction_summaries.csv", index=False)
    selection_manifest_df.to_csv(out_dir / "selection_manifest.csv", index=False)

    curve_summary_df = build_curve_level_summary(best_runs_df)
    low_fraction_df = build_low_fraction_summary(best_runs_df)
    curve_summary_df.to_csv(out_dir / "curve_level_summary.csv", index=False)
    low_fraction_df.to_csv(out_dir / "low_fraction_summary.csv", index=False)

    for direction in sorted(best_runs_df["direction"].dropna().unique().tolist()):
        plot_metric_by_fraction(
            best_runs_df,
            direction,
            plots_dir / f"{direction}_f1_attack_vs_fraction.png",
            "f1_attack",
            f"{direction} test attack F1 vs prefix fraction",
        )
        plot_metric_by_fraction(
            best_runs_df,
            direction,
            plots_dir / f"{direction}_recall_attack_vs_fraction.png",
            "recall_attack",
            f"{direction} test attack recall vs prefix fraction",
        )
        plot_metric_by_fraction(
            best_runs_df,
            direction,
            plots_dir / f"{direction}_f1_macro_vs_fraction.png",
            "f1_macro",
            f"{direction} test macro F1 vs prefix fraction",
        )
        plot_metric_by_fraction(
            best_runs_df,
            direction,
            plots_dir / f"{direction}_accuracy_vs_fraction.png",
            "accuracy",
            f"{direction} test accuracy vs prefix fraction",
        )

    iot23_artifacts = build_iot23_scenario_artifacts(selection_manifest_df, runs_dir, plots_dir)
    unsw_artifacts = build_unsw_attack_category_artifacts(selection_manifest_df, runs_dir, plots_dir)
    if not iot23_artifacts.empty:
        iot23_artifacts.to_csv(out_dir / "selected_iot23_scenario_artifacts.csv", index=False)
    if not unsw_artifacts.empty:
        unsw_artifacts.to_csv(out_dir / "selected_unsw_attack_category_artifacts.csv", index=False)

    selection_payload = {
        "runs_dir": str(runs_dir),
        "selected_runs": selection_manifest_df.to_dict(orient="records"),
        "curve_level_summary_rows": int(len(curve_summary_df)),
        "low_fraction_summary_rows": int(len(low_fraction_df)),
    }
    save_json(selection_payload, out_dir / "analysis_manifest.json")


if __name__ == "__main__":
    main()
