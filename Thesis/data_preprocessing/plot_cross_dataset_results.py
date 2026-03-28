from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_metric_bar(df: pd.DataFrame, metric: str, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    labels = [f"{row['model']} | {row['direction']}" for _, row in df.iterrows()]
    values = df[metric].values

    ax.bar(labels, values)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot cross-dataset evaluation results.")
    parser.add_argument("--summary_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    summary_csv = Path(args.summary_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_csv)

    plot_metric_bar(
        df,
        metric="f1_macro",
        out_path=out_dir / "cross_dataset_f1_macro.png",
        title="Cross-Dataset F1-Macro",
    )

    plot_metric_bar(
        df,
        metric="recall_attack",
        out_path=out_dir / "cross_dataset_recall_attack.png",
        title="Cross-Dataset Attack Recall",
    )

    plot_metric_bar(
        df,
        metric="f1_attack",
        out_path=out_dir / "cross_dataset_f1_attack.png",
        title="Cross-Dataset Attack F1",
    )

    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()