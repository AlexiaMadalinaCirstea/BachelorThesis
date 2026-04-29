from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRIC_COLUMNS = [
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "precision_attack",
    "recall_attack",
    "f1_attack",
]

RUN_PATTERN = re.compile(r"^outputs_(iot23|unsw)(?:_(mlp))?_exp(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze all in-domain early-detection RF/MLP runs and generate plots/CSVs."
    )
    parser.add_argument(
        "--runs_dir",
        default="early_detection/in_domain_early_detection",
        help="Directory containing outputs_* run folders.",
    )
    parser.add_argument(
        "--out_dir",
        default="early_detection/in_domain_early_detection/analysis_outputs",
        help="Directory for merged CSVs and plots.",
    )
    return parser.parse_args()


def save_json(obj: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True)


def parse_run_folder_name(name: str) -> dict | None:
    match = RUN_PATTERN.match(name)
    if not match:
        return None

    dataset, mlp_flag, exp_number = match.groups()
    model = "mlp" if mlp_flag else "rf"
    return {
        "dataset": dataset,
        "model": model,
        "exp_number": int(exp_number),
        "run_name": name,
    }


def read_run_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_overall_fraction_summary(run_dir: Path, run_meta: dict, run_config: dict) -> pd.DataFrame:
    summary_path = run_dir / "overall_fraction_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing overall fraction summary: {summary_path}")

    df = pd.read_csv(summary_path)
    for key, value in run_meta.items():
        df[key] = value

    df["model_label"] = df["model"].str.upper()
    df["dataset_label"] = df["dataset"].map({"iot23": "IoT-23", "unsw": "UNSW-NB15"})
    df["run_label"] = df["run_name"]
    df["train_rows"] = run_config.get("train_rows")
    df["val_rows_config"] = run_config.get("val_rows")
    df["test_rows_config"] = run_config.get("test_rows")
    df["hidden_layers"] = str(run_config.get("mlp_hidden_layers")) if df["model"].iloc[0] == "mlp" else ""
    df["max_iter"] = run_config.get("mlp_max_iter") if df["model"].iloc[0] == "mlp" else None
    df["batch_size"] = run_config.get("mlp_batch_size") if df["model"].iloc[0] == "mlp" else None
    df["n_estimators"] = run_config.get("rf_n_estimators") if df["model"].iloc[0] == "rf" else None
    df["max_depth"] = run_config.get("rf_max_depth") if df["model"].iloc[0] == "rf" else None
    return df


def load_first_true_positive_summary(run_dir: Path, run_meta: dict) -> pd.DataFrame:
    rows: list[dict] = []

    for split in ["val", "test"]:
        path = run_dir / split / "first_true_positive_fraction.csv"
        if not path.exists():
            continue

        df = pd.read_csv(path)
        if df.empty:
            continue

        metric_col = "first_true_positive_fraction"
        if metric_col not in df.columns:
            continue

        values = pd.to_numeric(df[metric_col], errors="coerce")
        valid_values = values.dropna()

        rows.append(
            {
                **run_meta,
                "split": split,
                "n_rows": int(len(df)),
                "n_valid_rows": int(valid_values.shape[0]),
                "first_true_positive_fraction_mean": float(valid_values.mean()) if not valid_values.empty else math.nan,
                "first_true_positive_fraction_median": float(valid_values.median()) if not valid_values.empty else math.nan,
                "first_true_positive_fraction_min": float(valid_values.min()) if not valid_values.empty else math.nan,
                "first_true_positive_fraction_max": float(valid_values.max()) if not valid_values.empty else math.nan,
            }
        )

    return pd.DataFrame(rows)


def collect_run_data(runs_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_summaries = []
    all_first_tp = []
    run_inventory = []

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        run_meta = parse_run_folder_name(run_dir.name)
        if run_meta is None:
            continue

        config_path = run_dir / "run_config.json"
        if not config_path.exists():
            continue

        run_config = read_run_config(config_path)
        summary_df = load_overall_fraction_summary(run_dir, run_meta, run_config)
        first_tp_df = load_first_true_positive_summary(run_dir, run_meta)

        inventory_row = {
            **run_meta,
            "train_rows": run_config.get("train_rows"),
            "val_rows": run_config.get("val_rows"),
            "test_rows": run_config.get("test_rows"),
            "hidden_layers": str(run_config.get("mlp_hidden_layers")) if run_meta["model"] == "mlp" else "",
            "max_iter": run_config.get("mlp_max_iter") if run_meta["model"] == "mlp" else None,
            "batch_size": run_config.get("mlp_batch_size") if run_meta["model"] == "mlp" else None,
            "n_estimators": run_config.get("rf_n_estimators") if run_meta["model"] == "rf" else None,
            "max_depth": run_config.get("rf_max_depth") if run_meta["model"] == "rf" else None,
        }

        all_summaries.append(summary_df)
        if not first_tp_df.empty:
            all_first_tp.append(first_tp_df)
        run_inventory.append(inventory_row)

    if not all_summaries:
        raise AssertionError(f"No valid run folders found in {runs_dir}")

    summary_all_df = pd.concat(all_summaries, ignore_index=True)
    first_tp_all_df = pd.concat(all_first_tp, ignore_index=True) if all_first_tp else pd.DataFrame()
    inventory_df = pd.DataFrame(run_inventory).sort_values(["dataset", "model", "exp_number"]).reset_index(drop=True)
    return summary_all_df, first_tp_all_df, inventory_df


def select_best_runs(summary_all_df: pd.DataFrame) -> pd.DataFrame:
    test_full = summary_all_df[
        (summary_all_df["split"] == "test") & (summary_all_df["fraction"] == 1.0)
    ].copy()
    if test_full.empty:
        raise AssertionError("No test fraction=1.0 rows found for best-run selection.")

    ranking = test_full.sort_values(
        ["dataset", "model", "f1_attack", "exp_number"],
        ascending=[True, True, False, False],
    )
    best_per_group = ranking.groupby(["dataset", "model"], as_index=False).first()
    best_run_names = set(best_per_group["run_name"].tolist())
    selected = summary_all_df[summary_all_df["run_name"].isin(best_run_names)].copy()
    return selected


def merge_first_tp(summary_df: pd.DataFrame, first_tp_df: pd.DataFrame) -> pd.DataFrame:
    if first_tp_df.empty:
        out = summary_df.copy()
        out["first_true_positive_fraction_mean"] = math.nan
        out["first_true_positive_fraction_median"] = math.nan
        return out

    return summary_df.merge(
        first_tp_df[
            [
                "dataset",
                "model",
                "run_name",
                "split",
                "first_true_positive_fraction_mean",
                "first_true_positive_fraction_median",
            ]
        ],
        on=["dataset", "model", "run_name", "split"],
        how="left",
    )


def plot_metric_by_fraction(df: pd.DataFrame, out_path: Path, metric: str, title: str) -> None:
    plt.figure(figsize=(10, 6))
    for dataset in ["iot23", "unsw"]:
        for model in ["rf", "mlp"]:
            subset = df[
                (df["dataset"] == dataset)
                & (df["model"] == model)
                & (df["split"] == "test")
            ].sort_values("fraction")
            if subset.empty:
                continue

            label = f"{subset['dataset_label'].iloc[0]} {subset['model_label'].iloc[0]}"
            plt.plot(
                subset["fraction"],
                subset[metric],
                marker="o",
                linewidth=2,
                label=label,
            )

    plt.title(title)
    plt.xlabel("Prefix Fraction")
    plt.ylabel(metric)
    plt.xticks(sorted(df["fraction"].dropna().unique().tolist()))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_first_tp(df: pd.DataFrame, out_path: Path) -> None:
    test_df = df[df["split"] == "test"].copy()
    if test_df.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    labels = []
    values = []

    for dataset in ["iot23", "unsw"]:
        for model in ["rf", "mlp"]:
            subset = test_df[(test_df["dataset"] == dataset) & (test_df["model"] == model)]
            if subset.empty:
                continue
            labels.append(f"{subset['dataset_label'].iloc[0]}\n{subset['model_label'].iloc[0]}")
            values.append(subset["first_true_positive_fraction_mean"].iloc[0])

    ax.bar(labels, values, color=["#4C78A8", "#F58518", "#54A24B", "#E45756"][: len(values)])
    ax.set_title("Mean First True Positive Fraction on Test Split")
    ax.set_ylabel("Mean Fraction")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_run_progression(df: pd.DataFrame, out_path: Path, dataset: str, model: str) -> None:
    subset = df[
        (df["dataset"] == dataset)
        & (df["model"] == model)
        & (df["split"] == "test")
        & (df["fraction"] == 1.0)
    ].sort_values("exp_number")
    if subset.empty:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(subset["exp_number"], subset["f1_attack"], marker="o", linewidth=2)
    plt.title(f"{subset['dataset_label'].iloc[0]} {subset['model_label'].iloc[0]} Run Progression at Full Fraction")
    plt.xlabel("Experiment Number")
    plt.ylabel("Test F1 Attack")
    plt.xticks(subset["exp_number"].tolist())
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def build_final_comparison_table(best_df: pd.DataFrame) -> pd.DataFrame:
    test_df = best_df[best_df["split"] == "test"].copy()
    columns = [
        "dataset_label",
        "model_label",
        "run_name",
        "fraction",
        "accuracy",
        "f1_macro",
        "f1_attack",
        "recall_attack",
        "precision_attack",
        "first_true_positive_fraction_mean",
    ]
    available_columns = [col for col in columns if col in test_df.columns]
    return test_df[available_columns].sort_values(["dataset_label", "model_label", "fraction"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    if args.runs_dir == "early_detection/in_domain_early_detection":
        runs_dir = script_dir
    else:
        runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        candidate = script_dir / args.runs_dir
        runs_dir = candidate if candidate.exists() else script_dir
    runs_dir = runs_dir.resolve()

    out_dir = Path(args.out_dir)
    if args.out_dir == "early_detection/in_domain_early_detection/analysis_outputs":
        out_dir = script_dir / "analysis_outputs"
    elif not out_dir.is_absolute():
        out_dir = (script_dir / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary_all_df, first_tp_all_df, inventory_df = collect_run_data(runs_dir)
    summary_all_with_tp_df = merge_first_tp(summary_all_df, first_tp_all_df)
    best_runs_df = select_best_runs(summary_all_with_tp_df)
    final_comparison_df = build_final_comparison_table(best_runs_df)

    inventory_df.to_csv(out_dir / "run_inventory.csv", index=False)
    summary_all_df.sort_values(["dataset", "model", "exp_number", "split", "fraction"]).to_csv(
        out_dir / "all_in_domain_runs_summary.csv",
        index=False,
    )
    summary_all_with_tp_df.sort_values(["dataset", "model", "exp_number", "split", "fraction"]).to_csv(
        out_dir / "all_in_domain_runs_with_first_tp.csv",
        index=False,
    )
    if not first_tp_all_df.empty:
        first_tp_all_df.sort_values(["dataset", "model", "exp_number", "split"]).to_csv(
            out_dir / "all_in_domain_first_true_positive_summary.csv",
            index=False,
        )
    best_runs_df.sort_values(["dataset", "model", "exp_number", "split", "fraction"]).to_csv(
        out_dir / "best_run_per_dataset_model.csv",
        index=False,
    )
    final_comparison_df.to_csv(out_dir / "final_in_domain_comparison.csv", index=False)

    plot_metric_by_fraction(
        best_runs_df,
        plots_dir / "f1_attack_vs_fraction.png",
        metric="f1_attack",
        title="Best In-Domain Runs: Test F1 Attack vs Prefix Fraction",
    )
    plot_metric_by_fraction(
        best_runs_df,
        plots_dir / "f1_macro_vs_fraction.png",
        metric="f1_macro",
        title="Best In-Domain Runs: Test F1 Macro vs Prefix Fraction",
    )
    plot_metric_by_fraction(
        best_runs_df,
        plots_dir / "recall_attack_vs_fraction.png",
        metric="recall_attack",
        title="Best In-Domain Runs: Test Recall Attack vs Prefix Fraction",
    )
    plot_metric_by_fraction(
        best_runs_df,
        plots_dir / "accuracy_vs_fraction.png",
        metric="accuracy",
        title="Best In-Domain Runs: Test Accuracy vs Prefix Fraction",
    )
    plot_first_tp(best_runs_df, plots_dir / "first_true_positive_comparison.png")

    for dataset in ["iot23", "unsw"]:
        for model in ["rf", "mlp"]:
            plot_run_progression(
                summary_all_with_tp_df,
                plots_dir / f"{dataset}_{model}_run_progression_f1_attack.png",
                dataset=dataset,
                model=model,
            )

    best_run_lookup = (
        best_runs_df[
            (best_runs_df["split"] == "test") & (best_runs_df["fraction"] == 1.0)
        ][["dataset", "model", "run_name", "f1_attack"]]
        .drop_duplicates()
        .sort_values(["dataset", "model"])
        .to_dict(orient="records")
    )
    save_json(
        {
            "runs_dir": str(runs_dir),
            "out_dir": str(out_dir),
            "n_total_runs": int(inventory_df.shape[0]),
            "selected_best_runs": best_run_lookup,
        },
        out_dir / "analysis_manifest.json",
    )


if __name__ == "__main__":
    main()
