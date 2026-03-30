from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


RANDOM_STATE = 42


# helpers


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {path} with shape {df.shape}")
    return df


# 1. feature alignment -> inspection of that

def inspect_alignment(
    csv_a: Path,
    csv_b: Path,
    label_col: str,
    drop_cols: list[str],
    out_dir: Path,
) -> None:
    df_a = load_csv(csv_a)
    df_b = load_csv(csv_b)

    forbidden = set([label_col] + drop_cols)

    cols_a = set(df_a.columns) - forbidden
    cols_b = set(df_b.columns) - forbidden

    shared = sorted(cols_a & cols_b)
    only_a = sorted(cols_a - cols_b)
    only_b = sorted(cols_b - cols_a)

    print("\n=== Alignment summary ===")
    print(f"Shared feature columns: {len(shared)}")
    print(f"Only in A: {len(only_a)}")
    print(f"Only in B: {len(only_b)}")

    print("\n=== First 30 shared columns ===")
    for c in shared[:30]:
        print(c)

    print("\n=== Columns only in A ===")
    for c in only_a[:50]:
        print(c)

    print("\n=== Columns only in B ===")
    for c in only_b[:50]:
        print(c)

    ensure_dir(out_dir)
    pd.DataFrame({"shared_features": shared}).to_csv(out_dir / "shared_features.csv", index=False)
    pd.DataFrame({"only_in_a": only_a}).to_csv(out_dir / "only_in_a.csv", index=False)
    pd.DataFrame({"only_in_b": only_b}).to_csv(out_dir / "only_in_b.csv", index=False)

    print(f"\nSaved alignment tables to: {out_dir}")


# feature stability plots


def load_stability_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "feature",
        "mean_importance",
        "feature_stability_index",
        "transfer_utility",
        "cv_importance",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    return df


def scatter_importance_vs_stability(
    df: pd.DataFrame,
    title: str,
    out_path: Path,
    top_k: int = 8,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.scatter(df["mean_importance"], df["feature_stability_index"], alpha=0.7)

    top_df = df.sort_values("transfer_utility", ascending=False).head(top_k)
    for _, row in top_df.iterrows():
        ax.annotate(
            row["feature"],
            (row["mean_importance"], row["feature_stability_index"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel("Mean normalized importance")
    ax.set_ylabel("Feature Stability Index (FSI)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def bar_top_transferable(
    df: pd.DataFrame,
    title: str,
    out_path: Path,
    top_k: int = 10,
) -> None:
    top_df = df.sort_values("transfer_utility", ascending=False).head(top_k).copy()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_df["feature"][::-1], top_df["transfer_utility"][::-1])
    ax.set_xlabel("Transfer Utility")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def bar_top_unstable(
    df: pd.DataFrame,
    title: str,
    out_path: Path,
    top_k: int = 10,
) -> None:
    top_df = df.sort_values("cv_importance", ascending=False).head(top_k).copy()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_df["feature"][::-1], top_df["cv_importance"][::-1])
    ax.set_xlabel("Coefficient of Variation of Importance")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def save_top_tables(df: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    df.sort_values("transfer_utility", ascending=False).head(10).to_csv(
        out_dir / f"{prefix}_top10_transferable_features.csv",
        index=False,
    )
    df.sort_values("feature_stability_index", ascending=False).head(10).to_csv(
        out_dir / f"{prefix}_top10_stable_features.csv",
        index=False,
    )
    df.sort_values("cv_importance", ascending=False).head(10).to_csv(
        out_dir / f"{prefix}_top10_unstable_features.csv",
        index=False,
    )


def generate_stability_plots(
    rf_summary: Path,
    xgb_summary: Path,
    out_dir: Path,
) -> None:
    ensure_dir(out_dir)

    rf_df = load_stability_summary(rf_summary)
    xgb_df = load_stability_summary(xgb_summary)

    scatter_importance_vs_stability(
        rf_df,
        "Random Forest: Importance vs Stability",
        out_dir / "rf_importance_vs_stability.png",
    )
    scatter_importance_vs_stability(
        xgb_df,
        "XGBoost: Importance vs Stability",
        out_dir / "xgb_importance_vs_stability.png",
    )

    bar_top_transferable(
        rf_df,
        "Random Forest: Top Transferable Features",
        out_dir / "rf_top_transferable_features.png",
    )
    bar_top_transferable(
        xgb_df,
        "XGBoost: Top Transferable Features",
        out_dir / "xgb_top_transferable_features.png",
    )

    bar_top_unstable(
        rf_df,
        "Random Forest: Top Unstable Features",
        out_dir / "rf_top_unstable_features.png",
    )
    bar_top_unstable(
        xgb_df,
        "XGBoost: Top Unstable Features",
        out_dir / "xgb_top_unstable_features.png",
    )

    save_top_tables(rf_df, out_dir, prefix="rf")
    save_top_tables(xgb_df, out_dir, prefix="xgb")

    print(f"Saved feature stability plots and tables to: {out_dir}")


# cross-dataset evaluation

def load_labeled_dataset(path: Path, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if label_col not in df.columns:
        raise ValueError(f"'{label_col}' column not found in {path}")
    print(f"Loaded {path} with shape {df.shape}")
    return df


def infer_feature_columns(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str,
    drop_cols: list[str],
) -> list[str]:
    forbidden = set([label_col] + drop_cols)

    train_cols = set(train_df.columns) - forbidden
    test_cols = set(test_df.columns) - forbidden

    common = sorted(train_cols & test_cols)
    if not common:
        raise ValueError("No shared features found between train and test datasets.")

    return common


def build_model(model_name: str):
    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )

    if model_name == "xgb":
        if XGBClassifier is None:
            raise ImportError("xgboost is not installed, but model 'xgb' was requested.")
        return XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    raise ValueError(f"Unsupported model: {model_name}")


def compute_metrics(y_true: pd.Series, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "precision_attack": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "recall_attack": float(
            recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "f1_attack": float(
            f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
    }


def evaluate_direction(
        
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_name: str,
    test_name: str,
    model_name: str,
    label_col: str,
    drop_cols: list[str],
    out_dir: Path,
) -> dict:
    import joblib
    features = infer_feature_columns(train_df, test_df, label_col, drop_cols)

    X_train = train_df[features].copy()
    y_train = train_df[label_col].copy()
    X_test = test_df[features].copy()
    y_test = test_df[label_col].copy()

    model = build_model(model_name)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    direction_dir = out_dir / f"{train_name}_to_{test_name}_{model_name}"
    ensure_dir(direction_dir)

    joblib.dump(model, direction_dir / "model.joblib")

    pd.DataFrame({"feature": features}).to_csv(
        direction_dir / "used_features.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "y_true": y_test.values,
            "y_pred": y_pred,
        }
    ).to_csv(direction_dir / "predictions.csv", index=False)

    pd.DataFrame(
        cm,
        index=["true_benign", "true_attack"],
        columns=["pred_benign", "pred_attack"],
    ).to_csv(direction_dir / "confusion_matrix.csv")

    with open(direction_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_dataset": train_name,
                "test_dataset": test_name,
                "model": model_name,
                "n_train": int(len(train_df)),
                "n_test": int(len(test_df)),
                "n_features": int(len(features)),
                "metrics": metrics,
            },
            f,
            indent=2,
        )

    return {
        "direction": f"{train_name}->{test_name}",
        "model": model_name,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_features": int(len(features)),
        **metrics,
    }


def run_cross_dataset_eval(
    iot_csv: Path,
    unsw_csv: Path,
    out_dir: Path,
    models: list[str],
    label_col: str,
    drop_cols: list[str],
) -> None:
    ensure_dir(out_dir)

    iot_df = load_labeled_dataset(iot_csv, label_col=label_col)
    unsw_df = load_labeled_dataset(unsw_csv, label_col=label_col)

    rows = []

    for model_name in models:
        rows.append(
            evaluate_direction(
                train_df=iot_df,
                test_df=unsw_df,
                train_name="iot23",
                test_name="unsw_nb15",
                model_name=model_name,
                label_col=label_col,
                drop_cols=drop_cols,
                out_dir=out_dir,
            )
        )

        rows.append(
            evaluate_direction(
                train_df=unsw_df,
                test_df=iot_df,
                train_name="unsw_nb15",
                test_name="iot23",
                model_name=model_name,
                label_col=label_col,
                drop_cols=drop_cols,
                out_dir=out_dir,
            )
        )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_dir / "cross_dataset_summary.csv", index=False)

    print("\nCross-dataset evaluation complete.\n")
    print(summary_df.to_string(index=False))
    print(f"\nSaved outputs to: {out_dir}")


# 4. cross-dataset plots

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


def generate_cross_dataset_plots(summary_csv: Path, out_dir: Path) -> None:
    ensure_dir(out_dir)

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

    print(f"Saved cross-dataset plots to: {out_dir}")



# cli

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Research analysis suite: alignment, stability plots, cross-dataset eval, and plots."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # align
    p_align = subparsers.add_parser("align", help="Inspect feature alignment between two CSV files.")
    p_align.add_argument("--csv_a", required=True)
    p_align.add_argument("--csv_b", required=True)
    p_align.add_argument("--label_col", default="label")
    p_align.add_argument("--drop_cols", nargs="*", default=[])
    p_align.add_argument("--out_dir", required=True)

    # stability_plots
    p_stability = subparsers.add_parser("stability_plots", help="Generate feature stability plots.")
    p_stability.add_argument("--rf_summary", required=True)
    p_stability.add_argument("--xgb_summary", required=True)
    p_stability.add_argument("--out_dir", required=True)

    # cross_eval
    p_cross_eval = subparsers.add_parser("cross_eval", help="Run cross-dataset evaluation.")
    p_cross_eval.add_argument("--iot_csv", required=True)
    p_cross_eval.add_argument("--unsw_csv", required=True)
    p_cross_eval.add_argument("--out_dir", required=True)
    p_cross_eval.add_argument("--models", nargs="+", default=["rf", "xgb"])
    p_cross_eval.add_argument("--label_col", default="label")
    p_cross_eval.add_argument("--drop_cols", nargs="*", default=[])

    # cross_plots
    p_cross_plots = subparsers.add_parser("cross_plots", help="Generate cross-dataset result plots.")
    p_cross_plots.add_argument("--summary_csv", required=True)
    p_cross_plots.add_argument("--out_dir", required=True)

    args = parser.parse_args()

    if args.command == "align":
        inspect_alignment(
            csv_a=Path(args.csv_a),
            csv_b=Path(args.csv_b),
            label_col=args.label_col,
            drop_cols=args.drop_cols,
            out_dir=Path(args.out_dir),
        )
    elif args.command == "stability_plots":
        generate_stability_plots(
            rf_summary=Path(args.rf_summary),
            xgb_summary=Path(args.xgb_summary),
            out_dir=Path(args.out_dir),
        )
    elif args.command == "cross_eval":
        run_cross_dataset_eval(
            iot_csv=Path(args.iot_csv),
            unsw_csv=Path(args.unsw_csv),
            out_dir=Path(args.out_dir),
            models=args.models,
            label_col=args.label_col,
            drop_cols=args.drop_cols,
        )
    elif args.command == "cross_plots":
        generate_cross_dataset_plots(
            summary_csv=Path(args.summary_csv),
            out_dir=Path(args.out_dir),
        )
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()