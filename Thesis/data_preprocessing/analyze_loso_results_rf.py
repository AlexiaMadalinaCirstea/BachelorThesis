from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(x: Any) -> float:
    if x is None:
        return np.nan
    try:
        return float(x)
    except Exception:
        return np.nan


def categorize_fold(row: pd.Series) -> str:
    f1 = row["f1_macro"]
    rec_att = row["recall_attack"]
    acc = row["accuracy"]

    if pd.isna(f1):
        return "unknown"
    if f1 >= 0.90:
        return "excellent"
    if f1 >= 0.75:
        return "strong"
    if f1 >= 0.55:
        return "unstable"
    if acc >= 0.95 and row["n_attack"] <= 1:
        return "trivial_or_single_class"
    return "failure"


def parse_fold_summary(path: Path) -> dict[str, Any]:
    d = load_json(path)

    out = {
        "fold_dir": str(path.parent),
        "scenario": d.get("scenario"),
        "fold_name": d.get("fold_name"),
        "n_rows": int(d.get("n_rows", 0)),
        "n_benign": int(d.get("n_benign", 0)),
        "n_attack": int(d.get("n_attack", 0)),
        "attack_ratio": (int(d.get("n_attack", 0)) / int(d.get("n_rows", 1))) if int(d.get("n_rows", 0)) > 0 else np.nan,
        "accuracy": safe_float(d.get("accuracy")),
        "balanced_accuracy": safe_float(d.get("balanced_accuracy")),
        "f1_macro": safe_float(d.get("f1_macro")),
        "f1_weighted": safe_float(d.get("f1_weighted")),
        "precision_macro": safe_float(d.get("precision_macro")),
        "recall_macro": safe_float(d.get("recall_macro")),
        "roc_auc": safe_float(d.get("roc_auc")),
        "pr_auc": safe_float(d.get("pr_auc")),
        "precision_benign": safe_float(d.get("precision_benign")),
        "recall_benign": safe_float(d.get("recall_benign")),
        "f1_benign": safe_float(d.get("f1_benign")),
        "support_benign": int(d.get("support_benign", 0)),
        "precision_attack": safe_float(d.get("precision_attack")),
        "recall_attack": safe_float(d.get("recall_attack")),
        "f1_attack": safe_float(d.get("f1_attack")),
        "support_attack": int(d.get("support_attack", 0)),
    }

    out["fnr_attack"] = np.nan if pd.isna(out["recall_attack"]) else 1.0 - out["recall_attack"]
    out["fpr_benign"] = np.nan if pd.isna(out["recall_benign"]) else 1.0 - out["recall_benign"]
    out["single_class_test"] = (out["n_benign"] == 0) or (out["n_attack"] == 0)
    out["dominant_class"] = "attack" if out["attack_ratio"] > 0.5 else "benign"
    out["category"] = categorize_fold(pd.Series(out))
    return out


def build_markdown_summary(df: pd.DataFrame, summary: dict[str, Any]) -> str:
    worst = df.sort_values("f1_macro", ascending=True).head(5)
    best = df.sort_values("f1_macro", ascending=False).head(5)
    high_fpr = df.sort_values("fpr_benign", ascending=False).head(5)
    high_fnr = df.sort_values("fnr_attack", ascending=False).head(5)

    def lines_from_rows(rows: pd.DataFrame, cols: list[str]) -> str:
        out = []
        for _, r in rows.iterrows():
            vals = ", ".join(
                f"{c}={r[c]:.4f}" if isinstance(r[c], (float, np.floating)) and not pd.isna(r[c]) else f"{c}={r[c]}"
                for c in cols
            )
            out.append(f"- {r['scenario']}: {vals}")
        return "\n".join(out)

    md = f"""# LOSO RF Analysis Summary

## Headline
- Number of folds: {summary['n_folds']}
- Total rows across folds: {summary['n_rows_total']}
- Unweighted mean F1-macro: {summary['mean_f1_macro']:.4f}
- Std F1-macro: {summary['std_f1_macro']:.4f}
- Weighted mean F1-macro: {summary['weighted_mean_f1_macro']:.4f}
- Unweighted mean attack recall: {summary['mean_recall_attack']:.4f}
- Std attack recall: {summary['std_recall_attack']:.4f}
- Weighted mean attack recall: {summary['weighted_mean_recall_attack']:.4f}
- Single-class folds: {summary['single_class_folds']}

## Interpretation
The Random Forest is not uniformly robust across unseen scenarios. 
Mean performance is lowered by severe fold-to-fold variance, indicating scenario 
dependence rather than stable cross-scenario generalization

## Worst folds by F1-macro
{lines_from_rows(worst, ['f1_macro', 'accuracy', 'recall_attack', 'recall_benign', 'attack_ratio'])}

## Best folds by F1-macro
{lines_from_rows(best, ['f1_macro', 'accuracy', 'recall_attack', 'recall_benign', 'attack_ratio'])}

## Highest benign false positive pressure
{lines_from_rows(high_fpr, ['fpr_benign', 'accuracy', 'f1_macro', 'attack_ratio'])}

## Highest attack miss rate
{lines_from_rows(high_fnr, ['fnr_attack', 'accuracy', 'f1_macro', 'attack_ratio'])}

## Category counts
{summary['category_counts']}

LOSO evaluation shows that strong pooled performance can hide substantial instability at the scenario level 
Some folds are solved almost perfectly, while others collapse due to class-skewed or shifted scenario distributions
This supports my thesis claim that IDS models must be evaluated under scenario-aware protocols rather than relying on a single aggregate split.
"""
    return md


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze LOSO fold_summary.json files")
    parser.add_argument("--loso_dir", required=True, help="Directory containing LOSO fold subfolders")
    parser.add_argument("--out_dir", default=None, help="Output directory for analysis files")
    args = parser.parse_args()

    loso_dir = Path(args.loso_dir)
    out_dir = Path(args.out_dir) if args.out_dir else loso_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_paths = sorted(loso_dir.glob("loso_*/fold_summary.json"))
    if not json_paths:
        raise FileNotFoundError(f"No fold_summary.json files found under {loso_dir}")

    rows = [parse_fold_summary(p) for p in json_paths]
    df = pd.DataFrame(rows).sort_values(["f1_macro", "scenario"], ascending=[True, True]).reset_index(drop=True)

    summary = {
        "n_folds": int(len(df)),
        "n_rows_total": int(df["n_rows"].sum()),
        "mean_accuracy": float(df["accuracy"].mean()),
        "std_accuracy": float(df["accuracy"].std(ddof=1)),
        "mean_f1_macro": float(df["f1_macro"].mean()),
        "std_f1_macro": float(df["f1_macro"].std(ddof=1)),
        "mean_f1_weighted": float(df["f1_weighted"].mean()),
        "std_f1_weighted": float(df["f1_weighted"].std(ddof=1)),
        "mean_recall_attack": float(df["recall_attack"].mean()),
        "std_recall_attack": float(df["recall_attack"].std(ddof=1)),
        "mean_recall_benign": float(df["recall_benign"].mean()),
        "std_recall_benign": float(df["recall_benign"].std(ddof=1)),
        "weighted_mean_f1_macro": float(np.average(df["f1_macro"], weights=df["n_rows"])),
        "weighted_mean_recall_attack": float(np.average(df["recall_attack"].fillna(0.0), weights=df["n_rows"])),
        "weighted_mean_recall_benign": float(np.average(df["recall_benign"].fillna(0.0), weights=df["n_rows"])),
        "single_class_folds": int(df["single_class_test"].sum()),
        "category_counts": df["category"].value_counts().to_dict(),
        "worst_fold_by_f1_macro": df.nsmallest(1, "f1_macro").iloc[0].to_dict(),
        "best_fold_by_f1_macro": df.nlargest(1, "f1_macro").iloc[0].to_dict(),
        "worst_fold_by_recall_attack": df.nsmallest(1, "recall_attack").iloc[0].to_dict(),
        "worst_fold_by_recall_benign": df.nsmallest(1, "recall_benign").iloc[0].to_dict(),
    }

    df.to_csv(out_dir / "all_fold_summaries.csv", index=False)
    df.sort_values("f1_macro", ascending=True).to_csv(out_dir / "ranked_by_f1_macro.csv", index=False)
    df.sort_values("recall_attack", ascending=True).to_csv(out_dir / "ranked_by_recall_attack.csv", index=False)
    df.sort_values("recall_benign", ascending=True).to_csv(out_dir / "ranked_by_recall_benign.csv", index=False)
    df.sort_values("attack_ratio", ascending=False).to_csv(out_dir / "ranked_by_attack_ratio.csv", index=False)

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    md = build_markdown_summary(df, summary)
    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write(md)

    print(f"Saved analysis to: {out_dir}")
    print(f"Mean F1-macro: {summary['mean_f1_macro']:.4f} ± {summary['std_f1_macro']:.4f}")
    print(f"Mean attack recall: {summary['mean_recall_attack']:.4f} ± {summary['std_recall_attack']:.4f}")
    print(f"Worst F1 fold: {summary['worst_fold_by_f1_macro']['scenario']}")
    print(f"Best F1 fold: {summary['best_fold_by_f1_macro']['scenario']}")


if __name__ == "__main__":
    main()