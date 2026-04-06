import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def plot_macro_f1_comparison(df: pd.DataFrame, out_dir: str):
    df = df.sort_values("macro_f1_rf_minus_xgb", ascending=False)

    attacks = df["held_out_attack"]
    rf_vals = df["macro_f1_rf"]
    xgb_vals = df["macro_f1_xgb"]

    x = range(len(attacks))
    width = 0.35

    plt.figure(figsize=(12, 6))

    plt.bar([i - width/2 for i in x], rf_vals, width=width, label="Random Forest")
    plt.bar([i + width/2 for i in x], xgb_vals, width=width, label="XGBoost")

    plt.xticks(x, attacks, rotation=45, ha="right")
    plt.ylabel("Macro-F1 Score")
    plt.title("Leave-One-Attack-Type-Out Performance (UNSW-NB15)")
    plt.legend()

    plt.tight_layout()

    out_path = os.path.join(out_dir, "macro_f1_per_attack.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved plot: {out_path}")


def plot_binary_f1_comparison(df: pd.DataFrame, out_dir: str):
    df = df.sort_values("binary_f1_rf_minus_xgb", ascending=False)

    attacks = df["held_out_attack"]
    rf_vals = df["binary_f1_rf"]
    xgb_vals = df["binary_f1_xgb"]

    x = range(len(attacks))
    width = 0.35

    plt.figure(figsize=(12, 6))

    plt.bar([i - width/2 for i in x], rf_vals, width=width, label="Random Forest")
    plt.bar([i + width/2 for i in x], xgb_vals, width=width, label="XGBoost")

    plt.xticks(x, attacks, rotation=45, ha="right")
    plt.ylabel("Binary F1 Score")
    plt.title("Binary F1 per Attack Type (UNSW-NB15)")
    plt.legend()

    plt.tight_layout()

    out_path = os.path.join(out_dir, "binary_f1_per_attack.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved plot: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.comparison_csv)

    plot_macro_f1_comparison(df, args.out_dir)
    plot_binary_f1_comparison(df, args.out_dir)


if __name__ == "__main__":
    main()