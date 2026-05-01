from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import pyarrow.parquet as pq
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


IOT23_NUMERIC_COLS = [
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

IOT23_CATEGORICAL_COLS = [
    "proto",
    "service",
    "conn_state",
]

UNSW_NUMERIC_COLS = [
    "dur",
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "rate",
    "sttl",
    "dttl",
    "sload",
    "dload",
    "sloss",
    "dloss",
    "sinpkt",
    "dinpkt",
    "sjit",
    "djit",
    "swin",
    "stcpb",
    "dtcpb",
    "dwin",
    "tcprtt",
    "synack",
    "ackdat",
    "smean",
    "dmean",
    "trans_depth",
    "response_body_len",
    "ct_srv_src",
    "ct_state_ttl",
    "ct_dst_ltm",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "ct_dst_src_ltm",
    "is_ftp_login",
    "ct_ftp_cmd",
    "ct_flw_http_mthd",
    "ct_src_ltm",
    "ct_srv_dst",
    "is_sm_ips_ports",
]

UNSW_CATEGORICAL_COLS = [
    "proto",
    "service",
    "state",
]

DEFAULT_FRACTIONS = [0.1, 0.2, 0.5, 1.0]
IOT23_REQUIRED_COLS = IOT23_NUMERIC_COLS + IOT23_CATEGORICAL_COLS + ["scenario", "split", "ts", "label_binary", "label_phase"]
UNSW_REQUIRED_COLS = UNSW_NUMERIC_COLS + UNSW_CATEGORICAL_COLS + ["id", "attack_cat", "label"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Shared in-domain early-detection MLP runner for IoT-23 and UNSW-NB15."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["iot23", "unsw"],
        help="Dataset adapter to use.",
    )
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=DEFAULT_FRACTIONS,
        help="Prefix fractions to evaluate.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Directory for outputs.",
    )
    parser.add_argument(
        "--mlp_hidden_layers",
        nargs="+",
        type=int,
        default=[128, 64],
        help="Hidden layer sizes for the MLP.",
    )
    parser.add_argument(
        "--mlp_alpha",
        type=float,
        default=0.0001,
        help="L2 penalty.",
    )
    parser.add_argument(
        "--mlp_max_iter",
        type=int,
        default=40,
        help="Maximum MLP iterations.",
    )
    parser.add_argument(
        "--mlp_batch_size",
        type=int,
        default=512,
        help="MLP batch size.",
    )

    parser.add_argument(
        "--iot_data_dir",
        default="Datasets/IoT23/processed_full/iot23",
        help="IoT-23 processed data directory.",
    )
    parser.add_argument(
        "--iot_train_max_rows",
        type=int,
        default=None,
        help="Optional cap on IoT-23 training rows.",
    )
    parser.add_argument(
        "--iot_eval_max_rows_per_scenario",
        type=int,
        default=None,
        help="Optional cap on IoT-23 earliest evaluation rows per scenario.",
    )

    parser.add_argument(
        "--unsw_train_csv",
        default=r"Datasets/UNSW-NB15/UNSW-NB15 dataset/CSV Files/Training and Testing Sets/UNSW_NB15_training-set.csv",
        help="UNSW-NB15 training CSV.",
    )
    parser.add_argument(
        "--unsw_test_csv",
        default=r"Datasets/UNSW-NB15/UNSW-NB15 dataset/CSV Files/Training and Testing Sets/UNSW_NB15_testing-set.csv",
        help="UNSW-NB15 testing CSV.",
    )
    parser.add_argument(
        "--unsw_val_fraction",
        type=float,
        default=0.2,
        help="Validation fraction carved from the training CSV.",
    )
    parser.add_argument(
        "--unsw_train_max_rows",
        type=int,
        default=None,
        help="Optional cap on UNSW training rows.",
    )
    parser.add_argument(
        "--unsw_eval_max_rows",
        type=int,
        default=None,
        help="Optional cap on UNSW evaluation rows.",
    )
    return parser.parse_args()


def save_json(obj: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True)


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


def build_pipeline(
    numeric_cols: list[str],
    categorical_cols: list[str],
    hidden_layers: tuple[int, ...],
    alpha: float,
    max_iter: int,
    batch_size: int,
    seed: int,
) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            (
                "ordinal",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        alpha=alpha,
        batch_size=batch_size,
        learning_rate_init=0.001,
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        random_state=seed,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def maybe_sample_rows(df: pd.DataFrame, max_rows: int | None, seed: int) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=seed).reset_index(drop=True)


def cap_unsw_eval_rows(df: pd.DataFrame, max_rows: int | None) -> pd.DataFrame:
    ordered = df.sort_values("id", kind="mergesort").reset_index(drop=True)
    if max_rows is None or len(ordered) <= max_rows:
        return ordered
    return ordered.head(max_rows).reset_index(drop=True)


def prefix_by_scenario(df: pd.DataFrame, fraction: float) -> pd.DataFrame:
    parts = []
    for _, group in df.groupby("scenario", sort=False):
        keep = max(1, int(len(group) * fraction))
        parts.append(group.iloc[:keep])
    return pd.concat(parts, ignore_index=True)


def summarize_iot23_scenarios(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario, group in pred_df.groupby("scenario", sort=False):
        metrics = compute_metrics(group["label_binary"], group["y_pred"])
        rows.append(
            {
                "scenario": scenario,
                "rows": int(len(group)),
                "attack_rows": int(group["label_binary"].sum()),
                "attack_rate": float(group["label_binary"].mean()),
                "first_ts": float(group["ts"].min()),
                "last_ts": float(group["ts"].max()),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def build_iot23_latency_table(full_predictions: pd.DataFrame, fractions: list[float]) -> pd.DataFrame:
    rows = []
    for scenario, group in full_predictions.groupby("scenario", sort=False):
        ordered = group.sort_values("ts", kind="mergesort").reset_index(drop=True)
        attack_total = int(ordered["label_binary"].sum())
        for fraction in fractions:
            keep = max(1, int(len(ordered) * fraction))
            prefix = ordered.iloc[:keep]
            rows.append(
                {
                    "scenario": scenario,
                    "fraction": fraction,
                    "rows_seen": int(keep),
                    "scenario_rows_total": int(len(ordered)),
                    "scenario_attack_rows_total": attack_total,
                    "prefix_has_true_positive": bool(
                        ((prefix["label_binary"] == 1) & (prefix["y_pred"] == 1)).any()
                    ),
                }
            )
    return pd.DataFrame(rows)


def first_true_positive_by_scenario(latency_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario, group in latency_df.groupby("scenario", sort=False):
        positive_rows = group[group["prefix_has_true_positive"]]
        first_fraction = None if positive_rows.empty else float(positive_rows.iloc[0]["fraction"])
        rows.append({"scenario": scenario, "first_true_positive_fraction": first_fraction})
    return pd.DataFrame(rows)


def load_iot23_split(data_dir: Path, split_name: str) -> pd.DataFrame:
    path = data_dir / f"{split_name}.parquet"
    return pd.read_parquet(path, columns=IOT23_REQUIRED_COLS)


def stream_sample_iot23_train(path: Path, max_rows: int, seed: int) -> pd.DataFrame:
    parquet = pq.ParquetFile(path)
    total_rows = parquet.metadata.num_rows
    if total_rows <= max_rows:
        return pd.read_parquet(path, columns=IOT23_REQUIRED_COLS)

    sample_prob = min(1.0, max_rows / total_rows)
    sampled_parts = []

    for batch_index, batch in enumerate(parquet.iter_batches(columns=IOT23_REQUIRED_COLS, batch_size=100_000)):
        batch_df = batch.to_pandas()
        sampled_batch = batch_df.sample(
            frac=sample_prob,
            random_state=seed + batch_index,
        )
        if not sampled_batch.empty:
            sampled_parts.append(sampled_batch)

    if not sampled_parts:
        return pd.read_parquet(path, columns=IOT23_REQUIRED_COLS).head(max_rows).reset_index(drop=True)

    sampled_df = pd.concat(sampled_parts, ignore_index=True)
    if len(sampled_df) > max_rows:
        sampled_df = sampled_df.sample(n=max_rows, random_state=seed).reset_index(drop=True)
    return sampled_df.reset_index(drop=True)


def load_iot23_eval_split(
    data_dir: Path,
    split_name: str,
    max_rows_per_scenario: int | None,
) -> pd.DataFrame:
    if max_rows_per_scenario is None:
        return load_iot23_split(data_dir, split_name).sort_values(["scenario", "ts"], kind="mergesort").reset_index(drop=True)
    ordered = load_iot23_split(data_dir, split_name).sort_values(
        ["scenario", "ts"], kind="mergesort"
    ).reset_index(drop=True)
    return (
        ordered.groupby("scenario", sort=False, group_keys=False)
        .head(max_rows_per_scenario)
        .reset_index(drop=True)
    )


def run_iot23(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.iot_data_dir)
    fractions = sorted(set(args.fractions))

    train_path = data_dir / "train.parquet"
    if args.iot_train_max_rows is None:
        train_df = load_iot23_split(data_dir, "train")
    else:
        train_df = stream_sample_iot23_train(train_path, args.iot_train_max_rows, args.seed)

    val_df = load_iot23_eval_split(data_dir, "val", args.iot_eval_max_rows_per_scenario)
    test_df = load_iot23_eval_split(data_dir, "test", args.iot_eval_max_rows_per_scenario)

    pipeline = build_pipeline(
        numeric_cols=IOT23_NUMERIC_COLS,
        categorical_cols=IOT23_CATEGORICAL_COLS,
        hidden_layers=tuple(args.mlp_hidden_layers),
        alpha=args.mlp_alpha,
        max_iter=args.mlp_max_iter,
        batch_size=args.mlp_batch_size,
        seed=args.seed,
    )
    pipeline.fit(train_df[IOT23_NUMERIC_COLS + IOT23_CATEGORICAL_COLS], train_df["label_binary"])
    joblib.dump(pipeline, out_dir / "mlp_pipeline.joblib")

    all_summaries = []
    all_scenarios = []
    for split_name, df in [("val", val_df), ("test", test_df)]:
        split_dir = out_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        full_df = prefix_by_scenario(df, 1.0).copy()
        X_full = full_df[IOT23_NUMERIC_COLS + IOT23_CATEGORICAL_COLS]
        full_df["y_score"] = pipeline.predict_proba(X_full)[:, 1]
        full_df["y_pred"] = pipeline.predict(X_full)
        latency_df = build_iot23_latency_table(full_df, fractions)
        first_tp_df = first_true_positive_by_scenario(latency_df)
        latency_df.to_csv(split_dir / "detection_latency_by_fraction.csv", index=False)
        first_tp_df.to_csv(split_dir / "first_true_positive_fraction.csv", index=False)

        scenario_tables = []
        summary_rows = []
        for fraction in fractions:
            pred_df = prefix_by_scenario(df, fraction).copy()
            X_eval = pred_df[IOT23_NUMERIC_COLS + IOT23_CATEGORICAL_COLS]
            pred_df["y_score"] = pipeline.predict_proba(X_eval)[:, 1]
            pred_df["y_pred"] = pipeline.predict(X_eval)

            scenario_df = summarize_iot23_scenarios(pred_df).merge(first_tp_df, on="scenario", how="left")
            fraction_label = str(fraction).replace(".", "p")
            pred_df.to_parquet(split_dir / f"predictions_frac_{fraction_label}.parquet", index=False)
            scenario_df.to_csv(split_dir / f"scenario_metrics_frac_{fraction_label}.csv", index=False)

            malicious_scenarios = latency_df[
                (latency_df["fraction"] == fraction) & (latency_df["scenario_attack_rows_total"] > 0)
            ]
            detection_rate = None if malicious_scenarios.empty else float(malicious_scenarios["prefix_has_true_positive"].mean())
            summary_rows.append(
                {
                    "split": split_name,
                    "fraction": fraction,
                    "rows_evaluated": int(len(pred_df)),
                    "n_scenarios": int(pred_df["scenario"].nunique()),
                    "malicious_scenarios_detected_rate": detection_rate,
                    **compute_metrics(pred_df["label_binary"], pred_df["y_pred"]),
                }
            )
            scenario_tables.append(scenario_df.assign(split=split_name, fraction=fraction))

        summary_df = pd.DataFrame(summary_rows)
        scenario_all_df = pd.concat(scenario_tables, ignore_index=True)
        summary_df.to_csv(split_dir / "fraction_summary.csv", index=False)
        scenario_all_df.to_csv(split_dir / "scenario_metrics_all_fractions.csv", index=False)
        all_summaries.append(summary_df)
        all_scenarios.append(scenario_all_df)

    pd.concat(all_summaries, ignore_index=True).to_csv(out_dir / "overall_fraction_summary.csv", index=False)
    pd.concat(all_scenarios, ignore_index=True).to_csv(out_dir / "overall_scenario_summary.csv", index=False)

    save_json(
        {
            "dataset": "iot23",
            "data_dir": str(data_dir),
            "fractions": fractions,
            "seed": args.seed,
            "mlp_hidden_layers": list(args.mlp_hidden_layers),
            "mlp_alpha": args.mlp_alpha,
            "mlp_max_iter": args.mlp_max_iter,
            "mlp_batch_size": args.mlp_batch_size,
            "iot_train_max_rows": args.iot_train_max_rows,
            "iot_eval_max_rows_per_scenario": args.iot_eval_max_rows_per_scenario,
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
        },
        out_dir / "run_config.json",
    )


def load_unsw_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=UNSW_REQUIRED_COLS)
    return df.sort_values("id", kind="mergesort").reset_index(drop=True)


def split_unsw_train_val(df: pd.DataFrame, val_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts = []
    val_parts = []
    for label_value, group in df.groupby("label", sort=False):
        n_val = int(round(len(group) * val_fraction))
        n_val = max(1, min(len(group) - 1, n_val))
        val_subset = group.sample(n=n_val, random_state=seed)
        train_subset = group.drop(val_subset.index)
        train_parts.append(train_subset)
        val_parts.append(val_subset)
    train_df = pd.concat(train_parts, ignore_index=False).sort_values("id", kind="mergesort").reset_index(drop=True)
    val_df = pd.concat(val_parts, ignore_index=False).sort_values("id", kind="mergesort").reset_index(drop=True)
    return train_df, val_df


def prefix_frame(df: pd.DataFrame, fraction: float) -> pd.DataFrame:
    keep = max(1, int(len(df) * fraction))
    return df.iloc[:keep].copy().reset_index(drop=True)


def summarize_attack_categories(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for category, group in pred_df.groupby("attack_cat", sort=False):
        metrics = compute_metrics(group["label"], group["y_pred"])
        rows.append(
            {
                "attack_cat": category,
                "rows": int(len(group)),
                "attack_rows": int(group["label"].sum()),
                "attack_rate": float(group["label"].mean()),
                "first_id": int(group["id"].min()),
                "last_id": int(group["id"].max()),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def first_true_positive_fraction_unsw(predictions_by_fraction: dict[float, pd.DataFrame]) -> float | None:
    for fraction in sorted(predictions_by_fraction):
        df = predictions_by_fraction[fraction]
        if ((df["label"] == 1) & (df["y_pred"] == 1)).any():
            return float(fraction)
    return None


def run_unsw(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fractions = sorted(set(args.fractions))

    train_full_df = load_unsw_csv(Path(args.unsw_train_csv))
    test_df = load_unsw_csv(Path(args.unsw_test_csv))
    train_df, val_df = split_unsw_train_val(train_full_df, args.unsw_val_fraction, args.seed)

    train_df = maybe_sample_rows(train_df, args.unsw_train_max_rows, args.seed)
    val_df = cap_unsw_eval_rows(val_df, args.unsw_eval_max_rows)
    test_df = cap_unsw_eval_rows(test_df, args.unsw_eval_max_rows)

    pipeline = build_pipeline(
        numeric_cols=UNSW_NUMERIC_COLS,
        categorical_cols=UNSW_CATEGORICAL_COLS,
        hidden_layers=tuple(args.mlp_hidden_layers),
        alpha=args.mlp_alpha,
        max_iter=args.mlp_max_iter,
        batch_size=args.mlp_batch_size,
        seed=args.seed,
    )
    pipeline.fit(train_df[UNSW_NUMERIC_COLS + UNSW_CATEGORICAL_COLS], train_df["label"])
    joblib.dump(pipeline, out_dir / "mlp_pipeline.joblib")

    all_summaries = []
    all_categories = []
    for split_name, df in [("val", val_df), ("test", test_df)]:
        split_dir = out_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        predictions_by_fraction: dict[float, pd.DataFrame] = {}
        for fraction in fractions:
            pred_df = prefix_frame(df, fraction)
            X_eval = pred_df[UNSW_NUMERIC_COLS + UNSW_CATEGORICAL_COLS]
            pred_df["y_score"] = pipeline.predict_proba(X_eval)[:, 1]
            pred_df["y_pred"] = pipeline.predict(X_eval)
            predictions_by_fraction[fraction] = pred_df

        first_tp = first_true_positive_fraction_unsw(predictions_by_fraction)
        pd.DataFrame([{"split": split_name, "first_true_positive_fraction": first_tp}]).to_csv(
            split_dir / "first_true_positive_fraction.csv",
            index=False,
        )

        summary_rows = []
        category_tables = []
        for fraction, pred_df in predictions_by_fraction.items():
            category_df = summarize_attack_categories(pred_df)
            category_df["split"] = split_name
            category_df["fraction"] = fraction
            category_df["first_true_positive_fraction"] = first_tp
            fraction_label = str(fraction).replace(".", "p")
            pred_df.to_parquet(split_dir / f"predictions_frac_{fraction_label}.parquet", index=False)
            category_df.to_csv(split_dir / f"attack_cat_metrics_frac_{fraction_label}.csv", index=False)
            summary_rows.append(
                {
                    "split": split_name,
                    "fraction": fraction,
                    "rows_evaluated": int(len(pred_df)),
                    "n_attack_categories": int(pred_df["attack_cat"].nunique()),
                    "first_true_positive_fraction": first_tp,
                    **compute_metrics(pred_df["label"], pred_df["y_pred"]),
                }
            )
            category_tables.append(category_df)

        summary_df = pd.DataFrame(summary_rows)
        category_all_df = pd.concat(category_tables, ignore_index=True)
        summary_df.to_csv(split_dir / "fraction_summary.csv", index=False)
        category_all_df.to_csv(split_dir / "attack_cat_metrics_all_fractions.csv", index=False)
        all_summaries.append(summary_df)
        all_categories.append(category_all_df)

    pd.concat(all_summaries, ignore_index=True).to_csv(out_dir / "overall_fraction_summary.csv", index=False)
    pd.concat(all_categories, ignore_index=True).to_csv(out_dir / "overall_attack_cat_summary.csv", index=False)

    save_json(
        {
            "dataset": "unsw",
            "train_csv": args.unsw_train_csv,
            "test_csv": args.unsw_test_csv,
            "fractions": fractions,
            "seed": args.seed,
            "mlp_hidden_layers": list(args.mlp_hidden_layers),
            "mlp_alpha": args.mlp_alpha,
            "mlp_max_iter": args.mlp_max_iter,
            "mlp_batch_size": args.mlp_batch_size,
            "unsw_val_fraction": args.unsw_val_fraction,
            "unsw_train_max_rows": args.unsw_train_max_rows,
            "unsw_eval_max_rows": args.unsw_eval_max_rows,
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "train_attack_rate": float(train_df["label"].mean()),
            "val_attack_rate": float(val_df["label"].mean()),
            "test_attack_rate": float(test_df["label"].mean()),
        },
        out_dir / "run_config.json",
    )


def main() -> None:
    args = parse_args()
    if args.dataset == "iot23":
        run_iot23(args)
    elif args.dataset == "unsw":
        run_unsw(args)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")


if __name__ == "__main__":
    main()
