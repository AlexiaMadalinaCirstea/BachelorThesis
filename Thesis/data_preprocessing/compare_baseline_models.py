import json
import pandas as pd
from pathlib import Path


def load_summary(path):
    with open(path, "r") as f:
        return json.load(f)


def extract_metrics(summary):
    return {
        "f1_pooled": summary["pooled_overall"]["f1_macro"],
        "f1_mean": summary["fold_mean"]["f1_macro"],
        "f1_std": summary["fold_std"]["f1_macro"],
        "recall_attack_mean": summary["fold_mean"]["recall_attack"],
        "recall_attack_std": summary["fold_std"]["recall_attack"],
        "worst_f1": summary["worst_fold_by_f1_macro"]["f1_macro"],
        "best_f1": summary["best_fold_by_f1_macro"]["f1_macro"],
    }


def main():
    rf_summary = load_summary("Datasets/IoT23/processed_test_sample/iot23/rf_loso/loso_summary.json")
    xgb_summary = load_summary("Datasets/IoT23/processed_test_sample/iot23/xgb_loso/loso_summary.json")

    rf = extract_metrics(rf_summary)
    xgb = extract_metrics(xgb_summary)

    df = pd.DataFrame([
        {
            "Model": "Random Forest",
            "F1 (pooled)": rf["f1_pooled"],
            "F1 (LOSO mean)": rf["f1_mean"],
            "F1 (LOSO std)": rf["f1_std"],
            "Recall Attack (mean)": rf["recall_attack_mean"],
            "Recall Attack (std)": rf["recall_attack_std"],
            "Worst F1": rf["worst_f1"],
            "Best F1": rf["best_f1"],
        },
        {
            "Model": "XGBoost",
            "F1 (pooled)": xgb["f1_pooled"],
            "F1 (LOSO mean)": xgb["f1_mean"],
            "F1 (LOSO std)": xgb["f1_std"],
            "Recall Attack (mean)": xgb["recall_attack_mean"],
            "Recall Attack (std)": xgb["recall_attack_std"],
            "Worst F1": xgb["worst_f1"],
            "Best F1": xgb["best_f1"],
        },
    ])

    print(df)

    out_path = Path("model_comparison.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved comparison to {out_path}")


if __name__ == "__main__":
    main()