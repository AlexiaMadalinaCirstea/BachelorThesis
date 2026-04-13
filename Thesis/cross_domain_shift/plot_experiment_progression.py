from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
FIGURES_DIR = BASE_DIR / "figures"
COMBINED_CSV = BASE_DIR / "combined_run_summary.csv"
RUN_NOTES_CSV = BASE_DIR / "combined_run_notes.csv"

METRICS = [
    ("f1_attack", "Attack F1"),
    ("recall_attack", "Attack Recall"),
    ("f1_macro", "Macro F1"),
]

RUN_TITLE_MAP = {
    "outputs": "R1",
    "outputs_2": "R2",
    "outputs_3": "R3",
    "outputs_4": "R4",
    "outputs_5": "R5",
}

MODEL_COLORS = {
    "rf": "#1b5e20",
    "xgb": "#c62828",
}

DIRECTION_TITLES = {
    "iot23_train->unsw_test": "IoT-23 -> UNSW-NB15",
    "unsw_train->iot23_test": "UNSW-NB15 -> IoT-23",
}


def natural_sort_key(path: Path) -> tuple[int, str]:
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
    return sorted(run_dirs, key=natural_sort_key)


def load_run_summary(run_dir: Path) -> pd.DataFrame:
    summary_path = run_dir / "cross_domain_shift_summary.csv"
    config_path = run_dir / "run_config.json"

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary CSV: {summary_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing run config JSON: {config_path}")

    summary_df = pd.read_csv(summary_path)
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    summary_df["run_folder"] = run_dir.name
    summary_df["run_id"] = RUN_TITLE_MAP.get(run_dir.name, run_dir.name)
    summary_df["include_review_features"] = bool(config.get("include_review_features", False))
    summary_df["aligned_feature_count"] = int(config.get("n_aligned_features", summary_df["n_features"].iloc[0]))
    summary_df["models_requested"] = ",".join(config.get("models", []))
    summary_df["rf_n_estimators"] = config.get("rf_n_estimators")
    summary_df["rf_max_depth"] = config.get("rf_max_depth")
    summary_df["xgb_n_estimators"] = config.get("xgb_n_estimators")
    summary_df["xgb_max_depth"] = config.get("xgb_max_depth")
    summary_df["train_row_cap_note"] = (
        f"IoT {config.get('iot_train_max_rows', 'full')} / "
        f"UNSW {config.get('unsw_train_max_rows', 'full')}"
    )
    summary_df["variant"] = (
        "Accepted 9-feature subset"
        if not summary_df["include_review_features"].iloc[0]
        else "Accepted subset + connection_state review feature"
    )
    summary_df["run_label"] = summary_df.apply(
        lambda row: (
            f"{row['run_id']}\n"
            f"{str(row['model']).upper()}\n"
            f"{int(row['aligned_feature_count'])}F"
        ),
        axis=1,
    )

    return summary_df


def build_combined_summary() -> pd.DataFrame:
    frames = [load_run_summary(run_dir) for run_dir in discover_run_dirs()]
    if not frames:
        raise FileNotFoundError("No outputs directories with summaries were found.")

    combined = pd.concat(frames, ignore_index=True)
    combined["run_sort"] = combined["run_folder"].map(
        {run_dir.name: natural_sort_key(run_dir)[0] for run_dir in discover_run_dirs()}
    )
    combined = combined.sort_values(["run_sort", "direction", "model"]).reset_index(drop=True)
    return combined


def save_run_notes(combined: pd.DataFrame) -> None:
    notes = (
        combined[
            [
                "run_id",
                "run_folder",
                "model",
                "aligned_feature_count",
                "include_review_features",
                "variant",
                "train_row_cap_note",
            ]
        ]
        .drop_duplicates()
        .rename(
            columns={
                "model": "primary_model",
                "aligned_feature_count": "feature_count",
                "include_review_features": "review_feature_included",
            }
        )
    )
    notes.to_csv(RUN_NOTES_CSV, index=False)


def annotate_bars(ax: plt.Axes, bars, precision: int = 3) -> None:
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.012,
            f"{value:.{precision}f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
            color="#1f1f1f",
        )


def format_axis(ax: plt.Axes, title: str, show_ylabel: bool) -> None:
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.22, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")
    if show_ylabel:
        ax.set_ylabel("Score", fontsize=10)


def build_plot(combined: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(17, 10),
        constrained_layout=False,
    )
    fig.patch.set_facecolor("#f7f4ed")

    for row_idx, direction in enumerate(DIRECTION_TITLES):
        direction_df = combined[combined["direction"] == direction].sort_values("run_sort")
        x = list(range(len(direction_df)))
        xtick_labels = direction_df["run_label"].tolist()

        for col_idx, (metric_key, metric_title) in enumerate(METRICS):
            ax = axes[row_idx, col_idx]
            values = direction_df[metric_key].tolist()
            colors = [MODEL_COLORS.get(model, "#455a64") for model in direction_df["model"]]
            hatches = ["//" if include else "" for include in direction_df["include_review_features"]]

            bars = ax.bar(
                x,
                values,
                color=colors,
                edgecolor="#263238",
                linewidth=0.8,
            )

            for bar, hatch in zip(bars, hatches):
                if hatch:
                    bar.set_hatch(hatch)

            annotate_bars(ax, bars)
            format_axis(
                ax=ax,
                title=f"{DIRECTION_TITLES[direction]} | {metric_title}",
                show_ylabel=(col_idx == 0),
            )
            ax.set_xticks(x)
            ax.set_xticklabels(xtick_labels, fontsize=9)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=MODEL_COLORS["rf"], edgecolor="#263238", label="Random Forest"),
        plt.Rectangle((0, 0), 1, 1, facecolor=MODEL_COLORS["xgb"], edgecolor="#263238", label="XGBoost"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#ffffff", edgecolor="#263238", hatch="//", label="Review feature included"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=3,
        frameon=False,
        fontsize=10,
    )

    fig.suptitle(
        "Cross-Domain Shift Progression Across IoT-23 and UNSW-NB15",
        fontsize=20,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.935,
        (
            "Runs R1-R5 track the staged progression from the first RF baseline to the larger XGB validation run. "
            "The main pattern stays stable: IoT-23 -> UNSW transfers partially, while UNSW -> IoT-23 remains near-total failure."
        ),
        ha="center",
        va="center",
        fontsize=11,
        color="#37474f",
    )
    fig.text(
        0.5,
        0.03,
        (
            "Run labels show run id, model, and aligned feature count. "
            "R4 is the only run that includes the review-feature ablation for connection_state/state."
        ),
        ha="center",
        fontsize=10,
        color="#455a64",
    )

    png_path = FIGURES_DIR / "cross_domain_shift_progression.png"
    svg_path = FIGURES_DIR / "cross_domain_shift_progression.svg"
    fig.tight_layout(rect=[0.03, 0.07, 0.97, 0.89])
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(svg_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    combined = build_combined_summary()
    combined.to_csv(COMBINED_CSV, index=False)
    save_run_notes(combined)
    build_plot(combined)

    print(f"Saved combined summary to: {COMBINED_CSV}")
    print(f"Saved run notes to: {RUN_NOTES_CSV}")
    print(f"Saved figures to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
