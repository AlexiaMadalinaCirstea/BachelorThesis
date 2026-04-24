from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import pyarrow.parquet as pq
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


NUMERIC_COLS = [
    "duration",
    "orig_bytes",
    "resp_bytes",
    "missed_bytes",
    "orig_pkts",
    "orig_ip_bytes",
    "resp_pkts",
    "resp_ip_bytes",
    "bytes_ratio",
    "pkts_ratio",
    "orig_bytes_per_pkt",
    "resp_bytes_per_pkt",
]

CATEGORICAL_COLS = [
    "proto",
    "service",
    "conn_state",
]

META_COLS = [
    "scenario",
    "split",
    "ts",
    "label_phase",
    "label_binary",
]

DEFAULT_FRACTIONS = [0.1, 0.2, 0.5, 1.0]
REQUIRED_COLS = list(dict.fromkeys(NUMERIC_COLS + CATEGORICAL_COLS + META_COLS))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="IoT-23 in-domain early detection using temporal scenario prefixes."
    )
    parser.add_argument(
        "--data_dir",
        default="Datasets/IoT23/processed_test_sample/iot23",
        help="Directory containing IoT-23 train/val/test parquet files.",
    )
    parser.add_argument(
        "--out_dir",
        default="early_detection/in_domain_early_detection/outputs_iot23",
        help="Directory for outputs.",
    )
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=DEFAULT_FRACTIONS,
        help="Temporal prefix fractions to evaluate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--target_col",
        default="label_binary",
        help="Target column name.",
    )
    parser.add_argument(
        "--train_max_rows",
        type=int,
        default=None,
        help="Optional random cap for training rows only.",
    )
    parser.add_argument(
        "--eval_max_rows_per_scenario",
        type=int,
        default=None,
        help="Optional cap on earliest rows kept per evaluation scenario after sorting by time.",
    )
    parser.add_argument(
        "--rf_n_estimators",
        type=int,
        default=300,
        help="Number of Random Forest trees.",
    )
    parser.add_argument(
        "--rf_max_depth",
        type=int,
        default=None,
        help="Optional Random Forest max depth.",
    )
    parser.add_argument(
        "--rf_n_jobs",
        type=int,
        default=1,
        help="Random Forest parallel jobs. Default 1 for sandbox-safe execution.",
    )
    return parser.parse_args()


def save_json(obj: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True)


def validate_columns(df: pd.DataFrame, target_col: str) -> None:
    required = set(NUMERIC_COLS + CATEGORICAL_COLS + META_COLS + [target_col])
    missing = required - set(df.columns)
    if missing:
        raise AssertionError(f"Missing required columns: {sorted(missing)}")


def load_split(data_dir: Path, split_name: str, target_col: str) -> pd.DataFrame:
    path = data_dir / f"{split_name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    df = pd.read_parquet(path, columns=REQUIRED_COLS)
    if df.empty:
        raise AssertionError(f"{split_name}.parquet is empty")
    validate_columns(df, target_col)
    return df


def stream_sample_train(path: Path, max_rows: int, seed: int) -> pd.DataFrame:
    parquet = pq.ParquetFile(path)
    total_rows = parquet.metadata.num_rows
    if total_rows <= max_rows:
        return pd.read_parquet(path, columns=REQUIRED_COLS)

    sample_prob = min(1.0, max_rows / total_rows)
    sampled_parts = []
    sampled_rows = 0

    for batch_index, batch in enumerate(parquet.iter_batches(columns=REQUIRED_COLS, batch_size=100_000)):
        batch_df = batch.to_pandas()
        sampled_batch = batch_df.sample(
            frac=sample_prob,
            random_state=seed + batch_index,
        )
        if not sampled_batch.empty:
            sampled_parts.append(sampled_batch)
            sampled_rows += len(sampled_batch)

    if not sampled_parts:
        fallback = pd.read_parquet(path, columns=REQUIRED_COLS).head(max_rows)
        return fallback.reset_index(drop=True)

    sampled_df = pd.concat(sampled_parts, ignore_index=True)
    if len(sampled_df) > max_rows:
        sampled_df = sampled_df.sample(n=max_rows, random_state=seed).reset_index(drop=True)
    return sampled_df.reset_index(drop=True)


def load_eval_split(
    data_dir: Path,
    split_name: str,
    target_col: str,
    max_rows_per_scenario: int | None,
) -> pd.DataFrame:
    path = data_dir / f"{split_name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")

    if max_rows_per_scenario is None:
        return load_split(data_dir, split_name, target_col)

    parquet = pq.ParquetFile(path)
    scenario_parts: dict[str, list[pd.DataFrame]] = {}
    scenario_counts: dict[str, int] = {}

    for batch in parquet.iter_batches(columns=REQUIRED_COLS, batch_size=100_000):
        batch_df = batch.to_pandas()
        for scenario, group in batch_df.groupby("scenario", sort=False):
            current = scenario_counts.get(scenario, 0)
            if current >= max_rows_per_scenario:
                continue

            keep = max_rows_per_scenario - current
            selected = group.head(keep)
            if selected.empty:
                continue

            scenario_parts.setdefault(scenario, []).append(selected)
            scenario_counts[scenario] = current + len(selected)

    if not scenario_parts:
        raise AssertionError(f"No rows collected for split {split_name}")

    collected = pd.concat(
        [pd.concat(parts, ignore_index=True) for parts in scenario_parts.values()],
        ignore_index=True,
    )
    validate_columns(collected, target_col)
    return prepare_eval_frame(collected, max_rows_per_scenario=None)


def prepare_eval_frame(df: pd.DataFrame, max_rows_per_scenario: int | None) -> pd.DataFrame:
    ordered = df.sort_values(["scenario", "ts"], kind="mergesort").reset_index(drop=True)
    if max_rows_per_scenario is None:
        return ordered

    limited = (
        ordered.groupby("scenario", sort=False, group_keys=False)
        .head(max_rows_per_scenario)
        .reset_index(drop=True)
    )
    return limited


def prefix_by_scenario(df: pd.DataFrame, fraction: float) -> pd.DataFrame:
    if not 0 < fraction <= 1:
        raise ValueError(f"Invalid fraction: {fraction}")

    parts = []
    for _, group in df.groupby("scenario", sort=False):
        n_rows = len(group)
        keep = max(1, int(n_rows * fraction))
        parts.append(group.iloc[:keep])

    return pd.concat(parts, ignore_index=True)


def build_pipeline(seed: int, n_estimators: int, max_depth: int | None, n_jobs: int) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="constant", fill_value=0.0))]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_COLS),
            ("cat", categorical_transformer, CATEGORICAL_COLS),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=seed,
        n_jobs=n_jobs,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_attack, recall_attack, f1_attack, support_attack = precision_recall_fscore_support(
        y_true, y_pred, labels=[1], average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_attack": float(precision_attack[0]),
        "recall_attack": float(recall_attack[0]),
        "f1_attack": float(f1_attack[0]),
        "attack_support": int(support_attack[0]),
        "false_negatives": int(cm[1, 0]),
        "false_positives": int(cm[0, 1]),
        "true_negatives": int(cm[0, 0]),
        "true_positives": int(cm[1, 1]),
    }


def summarize_scenarios(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario, group in pred_df.groupby("scenario", sort=False):
        y_true = group["label_binary"]
        y_pred = group["y_pred"]
        metrics = compute_metrics(y_true, y_pred)
        rows.append(
            {
                "scenario": scenario,
                "rows": int(len(group)),
                "attack_rows": int(y_true.sum()),
                "attack_rate": float(y_true.mean()),
                "first_ts": float(group["ts"].min()),
                "last_ts": float(group["ts"].max()),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def build_detection_latency_table(
    full_predictions: pd.DataFrame,
    fractions: list[float],
) -> pd.DataFrame:
    rows = []
    for scenario, group in full_predictions.groupby("scenario", sort=False):
        ordered = group.sort_values("ts", kind="mergesort").reset_index(drop=True)
        true_attack_rows = int(ordered["label_binary"].sum())
        for fraction in fractions:
            keep = max(1, int(len(ordered) * fraction))
            prefix = ordered.iloc[:keep]
            has_true_attack = bool(prefix["label_binary"].eq(1).any())
            has_predicted_attack = bool(prefix["y_pred"].eq(1).any())
            has_true_positive = bool(((prefix["label_binary"] == 1) & (prefix["y_pred"] == 1)).any())
            rows.append(
                {
                    "scenario": scenario,
                    "fraction": fraction,
                    "rows_seen": int(keep),
                    "scenario_rows_total": int(len(ordered)),
                    "scenario_attack_rows_total": true_attack_rows,
                    "prefix_has_true_attack": has_true_attack,
                    "prefix_has_predicted_attack": has_predicted_attack,
                    "prefix_has_true_positive": has_true_positive,
                }
            )
    return pd.DataFrame(rows)


def find_first_true_positive_fraction(latency_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario, group in latency_df.groupby("scenario", sort=False):
        positive_rows = group[group["prefix_has_true_positive"]]
        first_fraction = None if positive_rows.empty else float(positive_rows.iloc[0]["fraction"])
        rows.append(
            {
                "scenario": scenario,
                "first_true_positive_fraction": first_fraction,
            }
        )
    return pd.DataFrame(rows)


def evaluate_split(
    name: str,
    df: pd.DataFrame,
    pipeline: Pipeline,
    fractions: list[float],
    target_col: str,
    out_dir: Path,
) -> dict:
    split_dir = out_dir / name
    split_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    scenario_tables = []

    full_df = prefix_by_scenario(df, 1.0).copy()
    X_full = full_df[NUMERIC_COLS + CATEGORICAL_COLS].copy()
    full_df["y_score"] = pipeline.predict_proba(X_full)[:, 1]
    full_df["y_pred"] = pipeline.predict(X_full)

    latency_df = build_detection_latency_table(full_df, fractions)
    first_tp_df = find_first_true_positive_fraction(latency_df)
    latency_df.to_csv(split_dir / "detection_latency_by_fraction.csv", index=False)
    first_tp_df.to_csv(split_dir / "first_true_positive_fraction.csv", index=False)

    for fraction in fractions:
        prefix_df = prefix_by_scenario(df, fraction).copy()
        X_eval = prefix_df[NUMERIC_COLS + CATEGORICAL_COLS].copy()
        prefix_df["y_score"] = pipeline.predict_proba(X_eval)[:, 1]
        prefix_df["y_pred"] = pipeline.predict(X_eval)

        metrics = compute_metrics(prefix_df[target_col], prefix_df["y_pred"])
        scenario_summary = summarize_scenarios(prefix_df)
        merged_first_tp = scenario_summary.merge(first_tp_df, on="scenario", how="left")

        fraction_label = str(fraction).replace(".", "p")
        prefix_df.to_parquet(split_dir / f"predictions_frac_{fraction_label}.parquet", index=False)
        merged_first_tp.to_csv(split_dir / f"scenario_metrics_frac_{fraction_label}.csv", index=False)

        malicious_scenarios = latency_df[
            (latency_df["fraction"] == fraction) & (latency_df["scenario_attack_rows_total"] > 0)
        ]
        scenario_detection_rate = None
        if not malicious_scenarios.empty:
            scenario_detection_rate = float(malicious_scenarios["prefix_has_true_positive"].mean())

        summary_rows.append(
            {
                "split": name,
                "fraction": fraction,
                "rows_evaluated": int(len(prefix_df)),
                "n_scenarios": int(prefix_df["scenario"].nunique()),
                "malicious_scenarios_detected_rate": scenario_detection_rate,
                **metrics,
            }
        )
        scenario_tables.append(merged_first_tp.assign(split=name, fraction=fraction))

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(split_dir / "fraction_summary.csv", index=False)
    scenario_all_df = pd.concat(scenario_tables, ignore_index=True)
    scenario_all_df.to_csv(split_dir / "scenario_metrics_all_fractions.csv", index=False)

    return {
        "summary": summary_df,
        "scenarios": scenario_all_df,
    }


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fractions = sorted(set(args.fractions))

    train_path = data_dir / "train.parquet"
    if args.train_max_rows is None:
        train_df = load_split(data_dir, "train", args.target_col)
    else:
        train_df = stream_sample_train(train_path, args.train_max_rows, args.seed)
        validate_columns(train_df, args.target_col)

    val_df = load_eval_split(
        data_dir=data_dir,
        split_name="val",
        target_col=args.target_col,
        max_rows_per_scenario=args.eval_max_rows_per_scenario,
    )
    test_df = load_eval_split(
        data_dir=data_dir,
        split_name="test",
        target_col=args.target_col,
        max_rows_per_scenario=args.eval_max_rows_per_scenario,
    )

    pipeline = build_pipeline(
        seed=args.seed,
        n_estimators=args.rf_n_estimators,
        max_depth=args.rf_max_depth,
        n_jobs=args.rf_n_jobs,
    )

    X_train = train_df[NUMERIC_COLS + CATEGORICAL_COLS].copy()
    y_train = train_df[args.target_col].copy()
    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, out_dir / "rf_pipeline.joblib")

    val_results = evaluate_split(
        name="val",
        df=val_df,
        pipeline=pipeline,
        fractions=fractions,
        target_col=args.target_col,
        out_dir=out_dir,
    )
    test_results = evaluate_split(
        name="test",
        df=test_df,
        pipeline=pipeline,
        fractions=fractions,
        target_col=args.target_col,
        out_dir=out_dir,
    )

    combined_summary = pd.concat([val_results["summary"], test_results["summary"]], ignore_index=True)
    combined_summary.to_csv(out_dir / "overall_fraction_summary.csv", index=False)

    combined_scenarios = pd.concat([val_results["scenarios"], test_results["scenarios"]], ignore_index=True)
    combined_scenarios.to_csv(out_dir / "overall_scenario_summary.csv", index=False)

    run_config = {
        "data_dir": str(data_dir),
        "out_dir": str(out_dir),
        "fractions": fractions,
        "seed": args.seed,
        "target_col": args.target_col,
        "train_max_rows": args.train_max_rows,
        "eval_max_rows_per_scenario": args.eval_max_rows_per_scenario,
        "rf_n_estimators": args.rf_n_estimators,
        "rf_max_depth": args.rf_max_depth,
        "rf_n_jobs": args.rf_n_jobs,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "train_scenarios": sorted(train_df["scenario"].unique().tolist()),
        "val_scenarios": sorted(val_df["scenario"].unique().tolist()),
        "test_scenarios": sorted(test_df["scenario"].unique().tolist()),
    }
    save_json(run_config, out_dir / "run_config.json")


if __name__ == "__main__":
    main()
