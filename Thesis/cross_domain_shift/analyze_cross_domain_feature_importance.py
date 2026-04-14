from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "feature_importance_analysis"

RUN_LABELS = {
    "outputs": "R1",
    "outputs_2": "R2",
    "outputs_3": "R3",
    "outputs_4": "R4",
    "outputs_5": "R5",
}

DIRECTION_LABELS = {
    "iot23_train_to_unsw_test": "IoT-23 -> UNSW-NB15",
    "unsw_train_to_iot23_test": "UNSW-NB15 -> IoT-23",
}

MODEL_LABELS = {
    "rf": "Random Forest",
    "xgb": "XGBoost",
}


def natural_run_sort(path: Path) -> tuple[int, str]:
    if path.name == "outputs":
        return (1, path.name)

    match = re.fullmatch(r"outputs_(\d+)", path.name)
    if match:
        return (int(match.group(1)) + 1, path.name)

    return (9999, path.name)


def discover_run_dirs() -> list[Path]:
    run_dirs = [
        path for path in BASE_DIR.iterdir()
        if path.is_dir() and re.fullmatch(r"outputs(?:_\d+)?", path.name)
    ]
    return sorted(run_dirs, key=natural_run_sort)


def load_run_config(run_dir: Path) -> dict:
    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        return {}
    return pd.read_json(config_path, typ="series").to_dict()


def load_feature_importances() -> pd.DataFrame:
    rows: list[dict] = []

    for run_dir in discover_run_dirs():
        run_id = RUN_LABELS.get(run_dir.name, run_dir.name)
        config = load_run_config(run_dir)
        include_review = bool(config.get("include_review_features", False))
        feature_count = config.get("n_aligned_features")
        variant = (
            "Accepted 9-feature subset"
            if not include_review
            else "Accepted subset + connection_state"
        )

        for direction_dir in run_dir.iterdir():
            if not direction_dir.is_dir():
                continue
            if direction_dir.name not in DIRECTION_LABELS:
                continue

            for model_dir in direction_dir.iterdir():
                if not model_dir.is_dir():
                    continue

                importance_path = model_dir / "feature_importance.csv"
                if not importance_path.exists():
                    continue

                df = pd.read_csv(importance_path)
                df["run_folder"] = run_dir.name
                df["run_id"] = run_id
                df["direction"] = direction_dir.name
                df["direction_label"] = DIRECTION_LABELS[direction_dir.name]
                df["model"] = model_dir.name
                df["model_label"] = MODEL_LABELS.get(model_dir.name, model_dir.name.upper())
                df["include_review_features"] = include_review
                df["feature_count"] = feature_count
                df["variant"] = variant
                rows.extend(df.to_dict("records"))

    if not rows:
        raise FileNotFoundError("No feature_importance.csv files found in outputs folders.")

    combined = pd.DataFrame(rows)
    return combined


def save_top_feature_tables(df: pd.DataFrame) -> None:
    top10 = (
        df.sort_values(["run_id", "direction_label", "model_label", "importance"], ascending=[True, True, True, False])
        .groupby(["run_id", "direction_label", "model_label"], as_index=False)
        .head(10)
        .copy()
    )
    top10.to_csv(OUT_DIR / "top10_feature_importance_by_run.csv", index=False)

    average = (
        df.groupby(["feature", "direction_label", "model_label"], as_index=False)["importance"]
        .mean()
        .rename(columns={"importance": "mean_importance"})
        .sort_values(["direction_label", "model_label", "mean_importance"], ascending=[True, True, False])
    )
    average.to_csv(OUT_DIR / "mean_feature_importance_by_direction_model.csv", index=False)


def plot_direction_model_heatmaps(df: pd.DataFrame) -> None:
    for direction_label in sorted(df["direction_label"].unique()):
        for model_label in sorted(df["model_label"].unique()):
            subset = df[(df["direction_label"] == direction_label) & (df["model_label"] == model_label)].copy()
            if subset.empty:
                continue

            pivot = subset.pivot_table(
                index="feature",
                columns="run_id",
                values="importance",
                aggfunc="mean",
                fill_value=0.0,
            )

            pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

            plt.figure(figsize=(10, max(5, 0.45 * len(pivot))))
            sns.heatmap(
                pivot,
                annot=True,
                fmt=".3f",
                cmap="YlOrRd",
                linewidths=0.5,
                cbar_kws={"label": "Importance"},
            )
            plt.title(f"Feature Importance Heatmap\n{direction_label} | {model_label}", fontsize=14, weight="bold")
            plt.xlabel("Run")
            plt.ylabel("Aligned Feature")
            plt.tight_layout()

            safe_name = f"{direction_label}_{model_label}".lower()
            safe_name = re.sub(r"[^a-z0-9]+", "_", safe_name).strip("_")
            plt.savefig(OUT_DIR / f"{safe_name}_heatmap.png", dpi=300, bbox_inches="tight")
            plt.close()


def plot_average_importance(df: pd.DataFrame) -> None:
    avg_df = (
        df.groupby(["feature", "direction_label", "model_label"], as_index=False)["importance"]
        .mean()
        .rename(columns={"importance": "mean_importance"})
    )

    for direction_label in sorted(avg_df["direction_label"].unique()):
        subset = avg_df[avg_df["direction_label"] == direction_label].copy()
        if subset.empty:
            continue

        feature_order = (
            subset.groupby("feature")["mean_importance"]
            .mean()
            .sort_values(ascending=False)
            .index
        )

        plt.figure(figsize=(12, 7))
        sns.barplot(
            data=subset,
            x="mean_importance",
            y="feature",
            hue="model_label",
            order=feature_order,
            palette=["#1b5e20", "#c62828"],
        )
        plt.title(f"Average Cross-Domain Feature Importance\n{direction_label}", fontsize=15, weight="bold")
        plt.xlabel("Mean Importance Across Runs")
        plt.ylabel("Aligned Feature")
        plt.legend(title="Model", frameon=False)
        plt.tight_layout()

        safe_name = re.sub(r"[^a-z0-9]+", "_", direction_label.lower()).strip("_")
        plt.savefig(OUT_DIR / f"{safe_name}_average_importance.png", dpi=300, bbox_inches="tight")
        plt.close()


def plot_stability_scatter(df: pd.DataFrame) -> None:
    stability = (
        df.groupby(["feature", "direction_label", "model_label"], as_index=False)["importance"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "mean_importance", "std": "std_importance"})
    )
    stability["std_importance"] = stability["std_importance"].fillna(0.0)

    for direction_label in sorted(stability["direction_label"].unique()):
        subset = stability[stability["direction_label"] == direction_label].copy()
        if subset.empty:
            continue

        plt.figure(figsize=(10, 7))
        sns.scatterplot(
            data=subset,
            x="mean_importance",
            y="std_importance",
            hue="model_label",
            style="model_label",
            s=120,
            palette=["#1b5e20", "#c62828"],
        )

        for _, row in subset.iterrows():
            plt.text(
                row["mean_importance"] + 0.002,
                row["std_importance"] + 0.002,
                row["feature"],
                fontsize=8,
            )

        plt.title(f"Feature Importance Stability\n{direction_label}", fontsize=15, weight="bold")
        plt.xlabel("Mean Importance Across Runs")
        plt.ylabel("Std. Dev. Across Runs")
        plt.legend(title="Model", frameon=False)
        plt.tight_layout()

        safe_name = re.sub(r"[^a-z0-9]+", "_", direction_label.lower()).strip("_")
        plt.savefig(OUT_DIR / f"{safe_name}_stability_scatter.png", dpi=300, bbox_inches="tight")
        plt.close()

    stability.to_csv(OUT_DIR / "feature_importance_stability.csv", index=False)


def write_summary_note(df: pd.DataFrame) -> None:
    lines = []
    lines.append("# Cross-Domain Feature Importance Summary")
    lines.append("")
    lines.append("This file summarizes feature importance behavior across outputs through outputs_5.")
    lines.append("")

    avg = (
        df.groupby(["direction_label", "model_label", "feature"], as_index=False)["importance"]
        .mean()
        .rename(columns={"importance": "mean_importance"})
    )

    for direction_label in sorted(avg["direction_label"].unique()):
        lines.append(f"## {direction_label}")
        lines.append("")
        for model_label in sorted(avg["model_label"].unique()):
            subset = (
                avg[(avg["direction_label"] == direction_label) & (avg["model_label"] == model_label)]
                .sort_values("mean_importance", ascending=False)
                .head(5)
            )
            top_features = ", ".join(
                f"{row.feature} ({row.mean_importance:.3f})"
                for row in subset.itertuples()
            )
            lines.append(f"- {model_label}: {top_features}")
        lines.append("")

    (OUT_DIR / "FEATURE_IMPORTANCE_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_feature_importances()
    df.to_csv(OUT_DIR / "combined_feature_importance.csv", index=False)

    save_top_feature_tables(df)
    plot_direction_model_heatmaps(df)
    plot_average_importance(df)
    plot_stability_scatter(df)
    write_summary_note(df)

    print(f"Saved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
