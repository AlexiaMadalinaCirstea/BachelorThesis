from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


CATEGORICAL_COLS = [
    "proto",
    "service",
    "state",
]

NUMERIC_COLS = [
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

META_COLS = [
    "id",
    "attack_cat",
    "label",
]

DEFAULT_FRACTIONS = [0.1, 0.2, 0.5, 1.0]
TRAINING_FILE = r"Datasets/UNSW-NB15/UNSW-NB15 dataset/CSV Files/Training and Testing Sets/UNSW_NB15_training-set.csv"
TESTING_FILE = r"Datasets/UNSW-NB15/UNSW-NB15 dataset/CSV Files/Training and Testing Sets/UNSW_NB15_testing-set.csv"
REQUIRED_COLS = list(dict.fromkeys(CATEGORICAL_COLS + NUMERIC_COLS + META_COLS))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="UNSW-NB15 in-domain early detection using ordered flow prefixes."
    )
    parser.add_argument(
        "--train_csv",
        default=TRAINING_FILE,
        help="Path to UNSW-NB15 training CSV.",
    )
    parser.add_argument(
        "--test_csv",
        default=TESTING_FILE,
        help="Path to UNSW-NB15 testing CSV.",
    )
    parser.add_argument(
        "--out_dir",
        default="early_detection/in_domain_early_detection/outputs_unsw_exp1",
        help="Directory for outputs.",
    )
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=DEFAULT_FRACTIONS,
        help="Ordered prefix fractions to evaluate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.2,
        help="Fraction of the ordered training CSV reserved for validation.",
    )
    parser.add_argument(
        "--train_max_rows",
        type=int,
        default=None,
        help="Optional cap on training rows after carving validation off the training CSV.",
    )
    parser.add_argument(
        "--eval_max_rows",
        type=int,
        default=None,
        help="Optional cap on earliest validation and test rows kept for evaluation.",
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
        help="Random Forest parallel jobs. Default 1 for reliability.",
    )
    return parser.parse_args()


def save_json(obj: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True)


def validate_columns(df: pd.DataFrame) -> None:
    missing = set(REQUIRED_COLS) - set(df.columns)
    if missing:
        raise AssertionError(f"Missing required columns: {sorted(missing)}")


def load_unsw_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=REQUIRED_COLS)
    if df.empty:
        raise AssertionError(f"CSV is empty: {path}")
    validate_columns(df)
    return df.sort_values("id", kind="mergesort").reset_index(drop=True)


def split_train_val(df: pd.DataFrame, val_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1.")

    val_parts = []
    train_parts = []

    for label_value, group in df.groupby("label", sort=False):
        n_val = int(round(len(group) * val_fraction))
        if n_val <= 0 or n_val >= len(group):
            raise ValueError(
                f"val_fraction={val_fraction} creates an empty train or validation subset for label={label_value}."
            )

        val_subset = group.sample(n=n_val, random_state=seed)
        train_subset = group.drop(val_subset.index)

        val_parts.append(val_subset)
        train_parts.append(train_subset)

    train_df = (
        pd.concat(train_parts, ignore_index=False)
        .sort_values("id", kind="mergesort")
        .reset_index(drop=True)
    )
    val_df = (
        pd.concat(val_parts, ignore_index=False)
        .sort_values("id", kind="mergesort")
        .reset_index(drop=True)
    )
    return train_df, val_df


def maybe_sample_train(df: pd.DataFrame, max_rows: int | None, seed: int) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=seed).reset_index(drop=True)


def prepare_eval_frame(df: pd.DataFrame, max_rows: int | None) -> pd.DataFrame:
    ordered = df.sort_values("id", kind="mergesort").reset_index(drop=True)
    if max_rows is None or len(ordered) <= max_rows:
        return ordered
    return ordered.head(max_rows).reset_index(drop=True)


def prefix_frame(df: pd.DataFrame, fraction: float) -> pd.DataFrame:
    if not 0 < fraction <= 1:
        raise ValueError(f"Invalid fraction: {fraction}")
    keep = max(1, int(len(df) * fraction))
    return df.iloc[:keep].copy().reset_index(drop=True)


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


def attack_category_summary(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for category, group in pred_df.groupby("attack_cat", sort=False):
        y_true = group["label"]
        y_pred = group["y_pred"]
        metrics = compute_metrics(y_true, y_pred)
        rows.append(
            {
                "attack_cat": category,
                "rows": int(len(group)),
                "attack_rows": int(y_true.sum()),
                "attack_rate": float(y_true.mean()),
                "first_id": int(group["id"].min()),
                "last_id": int(group["id"].max()),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def first_true_positive_fraction(predictions_by_fraction: dict[float, pd.DataFrame]) -> float | None:
    for fraction in sorted(predictions_by_fraction):
        df = predictions_by_fraction[fraction]
        if ((df["label"] == 1) & (df["y_pred"] == 1)).any():
            return float(fraction)
    return None


def evaluate_split(
    split_name: str,
    df: pd.DataFrame,
    pipeline: Pipeline,
    fractions: list[float],
    out_dir: Path,
) -> dict:
    split_dir = out_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    predictions_by_fraction: dict[float, pd.DataFrame] = {}
    summary_rows = []
    category_tables = []

    for fraction in fractions:
        prefix_df = prefix_frame(df, fraction)
        X_eval = prefix_df[NUMERIC_COLS + CATEGORICAL_COLS].copy()
        prefix_df["y_score"] = pipeline.predict_proba(X_eval)[:, 1]
        prefix_df["y_pred"] = pipeline.predict(X_eval)
        predictions_by_fraction[fraction] = prefix_df

    first_tp = first_true_positive_fraction(predictions_by_fraction)
    first_tp_df = pd.DataFrame([{"split": split_name, "first_true_positive_fraction": first_tp}])
    first_tp_df.to_csv(split_dir / "first_true_positive_fraction.csv", index=False)

    for fraction, pred_df in predictions_by_fraction.items():
        metrics = compute_metrics(pred_df["label"], pred_df["y_pred"])
        category_df = attack_category_summary(pred_df)
        category_df["split"] = split_name
        category_df["fraction"] = fraction
        category_df["first_true_positive_fraction"] = first_tp
        category_tables.append(category_df)

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
                **metrics,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(split_dir / "fraction_summary.csv", index=False)

    combined_category_df = pd.concat(category_tables, ignore_index=True)
    combined_category_df.to_csv(split_dir / "attack_cat_metrics_all_fractions.csv", index=False)

    return {
        "summary": summary_df,
        "categories": combined_category_df,
    }


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_csv = Path(args.train_csv)
    test_csv = Path(args.test_csv)
    fractions = sorted(set(args.fractions))

    train_full_df = load_unsw_csv(train_csv)
    test_df = load_unsw_csv(test_csv)
    train_df, val_df = split_train_val(train_full_df, args.val_fraction, args.seed)

    train_df = maybe_sample_train(train_df, args.train_max_rows, args.seed)
    val_df = prepare_eval_frame(val_df, args.eval_max_rows)
    test_df = prepare_eval_frame(test_df, args.eval_max_rows)

    pipeline = build_pipeline(
        seed=args.seed,
        n_estimators=args.rf_n_estimators,
        max_depth=args.rf_max_depth,
        n_jobs=args.rf_n_jobs,
    )

    X_train = train_df[NUMERIC_COLS + CATEGORICAL_COLS].copy()
    y_train = train_df["label"].copy()
    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, out_dir / "rf_pipeline.joblib")

    val_results = evaluate_split("val", val_df, pipeline, fractions, out_dir)
    test_results = evaluate_split("test", test_df, pipeline, fractions, out_dir)

    pd.concat([val_results["summary"], test_results["summary"]], ignore_index=True).to_csv(
        out_dir / "overall_fraction_summary.csv",
        index=False,
    )
    pd.concat([val_results["categories"], test_results["categories"]], ignore_index=True).to_csv(
        out_dir / "overall_attack_cat_summary.csv",
        index=False,
    )

    run_config = {
        "train_csv": str(train_csv),
        "test_csv": str(test_csv),
        "out_dir": str(out_dir),
        "fractions": fractions,
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "train_max_rows": args.train_max_rows,
        "eval_max_rows": args.eval_max_rows,
        "rf_n_estimators": args.rf_n_estimators,
        "rf_max_depth": args.rf_max_depth,
        "rf_n_jobs": args.rf_n_jobs,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "train_attack_rate": float(train_df["label"].mean()),
        "val_attack_rate": float(val_df["label"].mean()),
        "test_attack_rate": float(test_df["label"].mean()),
    }
    save_json(run_config, out_dir / "run_config.json")


if __name__ == "__main__":
    main()
