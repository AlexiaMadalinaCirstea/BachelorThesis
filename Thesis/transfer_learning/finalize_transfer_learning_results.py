from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


FINAL_CONDITIONS = {
    "target_only_updated": "Target Only (Updated)",
    "transfer_learning_updated": "Transfer Learning (Updated)",
}

FINAL_DIRECTIONS = {
    "iot23_train->unsw_test": "IoT-23 -> UNSW-NB15",
    "unsw_train->iot23_test": "UNSW-NB15 -> IoT-23",
}

METRICS = ["f1_macro", "f1_attack", "recall_attack", "accuracy", "selected_threshold"]
MEANINGFUL_MACRO_F1_GAIN = 0.005


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create thesis-ready transfer-learning tables and a verification report "
            "from the updated-recipe outputs."
        )
    )
    parser.add_argument(
        "--summary_csv",
        default="transfer_learning/outputs_updated_recipe/updated_recipe_summary.csv",
        help="Path to updated_recipe_summary.csv.",
    )
    parser.add_argument(
        "--run_config",
        default="transfer_learning/outputs_updated_recipe/run_config.json",
        help="Path to the updated recipe run_config.json.",
    )
    parser.add_argument(
        "--outputs_dir",
        default="transfer_learning/outputs_updated_recipe",
        help="Base directory containing per-direction run outputs.",
    )
    parser.add_argument(
        "--out_dir",
        default="transfer_learning/final_tables",
        help="Directory for thesis-ready tables and verification output.",
    )
    return parser.parse_args()


def direction_slug(direction: str) -> str:
    mapping = {
        "iot23_train->unsw_test": "iot23_train_to_unsw_test",
        "unsw_train->iot23_test": "unsw_train_to_iot23_test",
    }
    return mapping[direction]


def condition_prefix(condition: str) -> str:
    mapping = {
        "target_only_updated": "target_only_updated",
        "transfer_learning_updated": "transfer_learning_updated",
    }
    return mapping[condition]


def fraction_slug(value: float) -> str:
    return f"frac_{str(value).replace('.', 'p')}"


def load_feature_importance(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_cols = {"feature", "importance"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df


def build_wide_results_table(df: pd.DataFrame) -> pd.DataFrame:
    long_df = df.copy()
    long_df["direction_label"] = long_df["direction"].map(FINAL_DIRECTIONS)
    long_df["condition_label"] = long_df["condition"].map(FINAL_CONDITIONS)

    wide = long_df.pivot_table(
        index=["direction_label", "target_fraction"],
        columns="condition_label",
        values=METRICS,
    )
    wide = wide.sort_index(axis=1, level=[0, 1])
    wide.columns = [f"{metric}_{condition}" for metric, condition in wide.columns]
    return wide.reset_index()


def build_gain_table(df: pd.DataFrame) -> pd.DataFrame:
    target_only = df[df["condition"] == "target_only_updated"].copy()
    target_only = target_only.rename(
        columns={
            "f1_macro": "target_only_f1_macro",
            "f1_attack": "target_only_f1_attack",
            "recall_attack": "target_only_recall_attack",
            "accuracy": "target_only_accuracy",
        }
    )

    transfer = df[df["condition"] == "transfer_learning_updated"].copy()
    merged = transfer.merge(
        target_only[
            [
                "direction",
                "target_fraction",
                "target_only_f1_macro",
                "target_only_f1_attack",
                "target_only_recall_attack",
                "target_only_accuracy",
            ]
        ],
        on=["direction", "target_fraction"],
        how="left",
    )
    merged["direction_label"] = merged["direction"].map(FINAL_DIRECTIONS)
    merged["macro_f1_gain_vs_target_only"] = merged["f1_macro"] - merged["target_only_f1_macro"]
    merged["attack_f1_gain_vs_target_only"] = merged["f1_attack"] - merged["target_only_f1_attack"]
    merged["attack_recall_gain_vs_target_only"] = (
        merged["recall_attack"] - merged["target_only_recall_attack"]
    )
    merged["accuracy_gain_vs_target_only"] = merged["accuracy"] - merged["target_only_accuracy"]
    return merged[
        [
            "direction_label",
            "target_fraction",
            "macro_f1_gain_vs_target_only",
            "attack_f1_gain_vs_target_only",
            "attack_recall_gain_vs_target_only",
            "accuracy_gain_vs_target_only",
        ]
    ].sort_values(["direction_label", "target_fraction"]).reset_index(drop=True)


def build_key_findings_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for direction, group in df.groupby("direction"):
        direction_label = FINAL_DIRECTIONS[direction]
        target = group[group["condition"] == "target_only_updated"].set_index("target_fraction")
        transfer = group[group["condition"] == "transfer_learning_updated"].set_index("target_fraction")
        gains = transfer["f1_macro"] - target["f1_macro"]
        positive = gains[gains > MEANINGFUL_MACRO_F1_GAIN]
        best_gain = float(gains.max())

        rows.append(
            {
                "direction": direction_label,
                "best_target_only_macro_f1": float(target["f1_macro"].max()),
                "best_transfer_macro_f1": float(transfer["f1_macro"].max()),
                "best_macro_f1_gain_vs_target_only": best_gain,
                "target_fractions_with_positive_macro_f1_gain": ",".join(
                    str(x) for x in positive.index.tolist()
                )
                if not positive.empty
                else "none",
                "overall_interpretation": (
                    "transfer helps under limited target data"
                    if not positive.empty
                    else "transfer does not meaningfully improve target-only"
                ),
            }
        )
    return pd.DataFrame(rows)


def build_feature_shift_table(outputs_dir: Path, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    transfer_rows = df[df["condition"] == "transfer_learning_updated"].copy()

    for _, row in transfer_rows.iterrows():
        run_dir = (
            outputs_dir
            / direction_slug(row["direction"])
            / f"{condition_prefix(row['condition'])}_{fraction_slug(row['target_fraction'])}"
        )
        pretrain_path = run_dir / "feature_importance_pretrain.csv"
        adapted_path = run_dir / "feature_importance.csv"
        if not pretrain_path.exists() or not adapted_path.exists():
            continue

        pretrain_df = load_feature_importance(pretrain_path).rename(
            columns={"importance": "pretrain_importance"}
        )
        adapted_df = load_feature_importance(adapted_path).rename(
            columns={"importance": "adapted_importance"}
        )
        merged = pretrain_df.merge(adapted_df, on="feature", how="outer").fillna(0.0)
        merged["absolute_shift"] = (merged["adapted_importance"] - merged["pretrain_importance"]).abs()
        merged = merged.sort_values("absolute_shift", ascending=False).head(3)

        for _, feat_row in merged.iterrows():
            rows.append(
                {
                    "direction": FINAL_DIRECTIONS[row["direction"]],
                    "target_fraction": row["target_fraction"],
                    "feature": feat_row["feature"],
                    "pretrain_importance": float(feat_row["pretrain_importance"]),
                    "adapted_importance": float(feat_row["adapted_importance"]),
                    "absolute_shift": float(feat_row["absolute_shift"]),
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["direction", "target_fraction", "absolute_shift"],
        ascending=[True, True, False],
    )


def build_verification_report(
    df: pd.DataFrame,
    outputs_dir: Path,
    run_config: dict[str, object],
) -> dict[str, object]:
    expected_directions = sorted(FINAL_DIRECTIONS.keys())
    expected_conditions = sorted(FINAL_CONDITIONS.keys())
    expected_fractions = list(run_config["target_fractions"])

    missing_runs = []
    for direction in expected_directions:
        for condition in expected_conditions:
            for fraction in expected_fractions:
                row_exists = (
                    (df["direction"] == direction)
                    & (df["condition"] == condition)
                    & (df["target_fraction"] == fraction)
                ).any()
                if not row_exists:
                    missing_runs.append(
                        {
                            "direction": direction,
                            "condition": condition,
                            "target_fraction": fraction,
                            "reason": "missing_from_summary_csv",
                        }
                    )
                    continue

                run_dir = (
                    outputs_dir
                    / direction_slug(direction)
                    / f"{condition_prefix(condition)}_{fraction_slug(fraction)}"
                )
                required_files = [
                    run_dir / "metrics.json",
                    run_dir / "predictions.csv",
                    run_dir / "confusion_matrix.csv",
                    run_dir / "feature_importance.csv",
                    run_dir / "threshold_summary.csv",
                ]
                if condition == "transfer_learning_updated":
                    required_files.append(run_dir / "feature_importance_pretrain.csv")

                for required_file in required_files:
                    if not required_file.exists():
                        missing_runs.append(
                            {
                                "direction": direction,
                                "condition": condition,
                                "target_fraction": fraction,
                                "reason": f"missing_file:{required_file.name}",
                            }
                        )

    return {
        "summary_rows": int(len(df)),
        "directions_found": sorted(df["direction"].unique().tolist()),
        "conditions_found": sorted(df["condition"].unique().tolist()),
        "target_fractions_found": sorted(df["target_fraction"].unique().tolist()),
        "missing_or_incomplete_runs": missing_runs,
        "all_expected_runs_present": len(missing_runs) == 0,
        "n_aligned_features": run_config.get("n_aligned_features"),
        "aligned_features": run_config.get("aligned_features", []),
    }


def main() -> None:
    args = parse_args()

    summary_path = Path(args.summary_csv)
    outputs_dir = Path(args.outputs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_path)
    with open(args.run_config, "r", encoding="utf-8") as handle:
        run_config = json.load(handle)

    required_cols = {
        "direction",
        "condition",
        "target_fraction",
        "f1_macro",
        "f1_attack",
        "recall_attack",
        "accuracy",
        "selected_threshold",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Summary CSV is missing required columns: {missing}")

    df = df[df["condition"].isin(FINAL_CONDITIONS)].copy()
    df["target_fraction"] = pd.to_numeric(df["target_fraction"], errors="raise")

    wide_table = build_wide_results_table(df)
    gain_table = build_gain_table(df)
    key_findings_table = build_key_findings_table(df)
    feature_shift_table = build_feature_shift_table(outputs_dir=outputs_dir, df=df)
    verification_report = build_verification_report(
        df=df,
        outputs_dir=outputs_dir,
        run_config=run_config,
    )

    wide_table.to_csv(out_dir / "transfer_learning_main_results_wide.csv", index=False)
    gain_table.to_csv(out_dir / "transfer_learning_gain_vs_target_only.csv", index=False)
    key_findings_table.to_csv(out_dir / "transfer_learning_key_findings.csv", index=False)
    feature_shift_table.to_csv(out_dir / "transfer_learning_feature_shift_top3.csv", index=False)

    with open(out_dir / "transfer_learning_verification_report.json", "w", encoding="utf-8") as handle:
        json.dump(verification_report, handle, indent=2)

    print("Saved final transfer-learning tables and verification report.")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
