from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
TRANSFER_DIR = SCRIPT_DIR.parent
if str(TRANSFER_DIR) not in sys.path:
    sys.path.insert(0, str(TRANSFER_DIR))

import run_transfer_learning as base
import transfer_learning_updated_recipe as updated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run pairwise transfer-learning hypothesis experiments over generated pseudo-domain pairs. "
            "For each pair and seed, the script evaluates source-only, target-only, and transfer-learning."
        )
    )
    parser.add_argument(
        "--domain_registry",
        default="transfer_learning/hypothesis/generated_pairs_heavier/domain_registry.csv",
    )
    parser.add_argument(
        "--domain_pairs",
        default="transfer_learning/hypothesis/generated_pairs_heavier/domain_pairs.csv",
    )
    parser.add_argument(
        "--alignment_csv",
        default="feature_alignment/comparison_outputs/aligned_features_curated.csv",
    )
    parser.add_argument(
        "--iot_scenario_dir",
        default="Datasets/IoT23/processed_full/iot23/processed_scenarios",
    )
    parser.add_argument(
        "--out_dir",
        default="transfer_learning/hypothesis/pairwise_runs",
    )
    parser.add_argument("--include_review_features", action="store_true")
    parser.add_argument(
        "--pair_limit",
        type=int,
        default=None,
        help="Optional cap on how many pair definitions to execute.",
    )
    parser.add_argument(
        "--pair_family",
        choices=["iot23_to_unsw", "unsw_to_iot23"],
        default=None,
        help="Run only one direction family if desired.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 2026],
        help="Random seeds used for repeated runs.",
    )
    parser.add_argument(
        "--target_fractions",
        nargs="+",
        type=float,
        default=[0.05, 0.10, 0.25, 0.50, 1.0],
    )
    parser.add_argument(
        "--source_max_rows",
        type=int,
        default=50000,
        help="Cap for each source-train domain after filtering.",
    )
    parser.add_argument(
        "--target_train_max_rows",
        type=int,
        default=30000,
        help="Cap for each resolved target-train pool after filtering.",
    )
    parser.add_argument(
        "--target_test_max_rows",
        type=int,
        default=20000,
        help="Cap for each target-test domain after filtering.",
    )
    parser.add_argument(
        "--balance_target_train",
        action="store_true",
        help="Apply binary balancing before calibration splitting.",
    )
    parser.add_argument(
        "--target_balance_ratio",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--calibration_fraction",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
    )
    parser.add_argument("--pretrain_estimators", type=int, default=50)
    parser.add_argument("--adapt_estimators", type=int, default=150)
    parser.add_argument("--target_only_estimators", type=int, default=150)
    parser.add_argument("--xgb_max_depth", type=int, default=6)
    parser.add_argument("--xgb_learning_rate", type=float, default=0.05)
    return parser.parse_args()


def slugify(value: object, max_len: int = 120) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if len(text) > max_len:
        text = text[:max_len].rstrip("_")
    return text or "item"


def load_registry(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "domain_id",
        "dataset",
        "grouping",
        "group_value",
        "domain_family",
        "split",
        "source_path",
        "n_rows",
        "n_benign",
        "n_attack",
        "filter_json",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Domain registry is missing columns: {sorted(missing)}")
    return df


def load_pairs(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "pair_id",
        "pair_family",
        "source_domain_id",
        "target_domain_id",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Domain pairs file is missing columns: {sorted(missing)}")
    return df


def make_args_for_seed(args: argparse.Namespace, seed: int) -> argparse.Namespace:
    seeded_args = copy.deepcopy(args)
    seeded_args.random_state = seed
    return seeded_args


def aligned_feature_lists(alignment_df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    iot_native = sorted(set(alignment_df["iot23_feature"].tolist() + ["label"]))
    unsw_native = sorted(set(alignment_df["unsw_feature"].tolist() + ["label"]))
    aligned_features = [col for col in alignment_df["aligned_feature"].tolist()]
    return iot_native, unsw_native, aligned_features


def align_single_domain(df: pd.DataFrame, dataset: str, alignment_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if dataset == "iot23":
        mapping = dict(zip(alignment_df["iot23_feature"], alignment_df["aligned_feature"]))
    elif dataset == "unsw":
        mapping = dict(zip(alignment_df["unsw_feature"], alignment_df["aligned_feature"]))
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    missing = [col for col in mapping.keys() if col not in df.columns]
    if missing:
        raise ValueError(f"{dataset} subset is missing aligned columns: {missing}")

    aligned = df[list(mapping.keys()) + ["label"]].rename(columns=mapping).copy()
    feature_cols = [col for col in alignment_df["aligned_feature"].tolist() if col in aligned.columns]
    return aligned, feature_cols


def allocate_caps(domain_rows: pd.DataFrame, total_cap: int | None) -> dict[str, int | None]:
    domain_ids = domain_rows["domain_id"].tolist()
    if total_cap is None or len(domain_ids) == 1:
        return {domain_id: total_cap for domain_id in domain_ids}

    weights = domain_rows["n_rows"].astype(float)
    weight_sum = float(weights.sum())
    if weight_sum <= 0:
        equal_cap = max(1, total_cap // len(domain_ids))
        return {domain_id: equal_cap for domain_id in domain_ids}

    raw_caps = total_cap * (weights / weight_sum)
    caps = [max(1, int(math.floor(value))) for value in raw_caps.tolist()]
    remaining = total_cap - sum(caps)

    fractional = [
        (raw_caps.iloc[idx] - caps[idx], idx)
        for idx in range(len(caps))
    ]
    fractional.sort(reverse=True)
    for _, idx in fractional[: max(0, remaining)]:
        caps[idx] += 1

    return {domain_id: cap for domain_id, cap in zip(domain_ids, caps)}


def load_iot_domain(
    domain_row: pd.Series,
    iot_native_cols: list[str],
    alignment_df: pd.DataFrame,
    scenario_dir: Path,
    max_rows: int | None,
    seed: int,
) -> tuple[pd.DataFrame, list[str]]:
    filter_spec = json.loads(domain_row["filter_json"])
    filters = filter_spec.get("filters", [])
    if len(filters) != 1:
        raise ValueError(f"Expected exactly one IoT filter, got: {filters}")

    filter_entry = filters[0]
    op = filter_entry.get("op")
    if op == "eq":
        scenario_names = [str(filter_entry["value"])]
    elif op == "in":
        scenario_names = [str(value) for value in filter_entry.get("values", [])]
    else:
        raise ValueError(f"Unsupported IoT filter op: {op}")

    native_parts: list[pd.DataFrame] = []
    per_scenario_cap = None
    if max_rows is not None and scenario_names:
        per_scenario_cap = max(1, int(math.ceil(max_rows / len(scenario_names))))

    for scenario_name in scenario_names:
        scenario_file = scenario_dir / f"{scenario_name}.parquet"
        if not scenario_file.exists():
            raise FileNotFoundError(f"IoT scenario parquet not found: {scenario_file}")

        native_parts.append(
            base.load_iot23(
                path=scenario_file,
                columns=iot_native_cols,
                max_rows=per_scenario_cap,
                random_state=seed,
            )
        )

    native_df = pd.concat(native_parts, ignore_index=True)
    native_df = base.maybe_sample_rows(native_df, max_rows=max_rows, random_state=seed)
    return align_single_domain(native_df, dataset="iot23", alignment_df=alignment_df)


def load_unsw_filtered_native(
    domain_row: pd.Series,
    unsw_native_cols: list[str],
    max_rows: int | None,
    seed: int,
) -> pd.DataFrame:
    filter_spec = json.loads(domain_row["filter_json"])
    source_path = Path(str(filter_spec["path"]))
    filters = filter_spec.get("filters", [])
    if len(filters) != 1:
        raise ValueError(f"Expected exactly one UNSW filter, got: {filters}")

    filter_col = filters[0]["column"]
    filter_value = str(filters[0]["value"])
    usecols = sorted(set(unsw_native_cols + [filter_col]))

    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(source_path, usecols=usecols, chunksize=200000):
        local = chunk.copy()
        local["label"] = base.normalize_binary_labels(local["label"])
        filter_series = local[filter_col].astype("string").fillna("missing").str.strip()
        matched = local[filter_series == filter_value]
        if not matched.empty:
            parts.append(matched[unsw_native_cols].copy())

    if not parts:
        raise ValueError(f"No rows matched UNSW domain filter for {domain_row['domain_id']}")

    native_df = pd.concat(parts, ignore_index=True)
    native_df = base.maybe_sample_rows(native_df, max_rows=max_rows, random_state=seed)
    return base.downcast_numeric_columns(native_df)


def load_unsw_domain(
    domain_row: pd.Series,
    unsw_native_cols: list[str],
    alignment_df: pd.DataFrame,
    max_rows: int | None,
    seed: int,
) -> tuple[pd.DataFrame, list[str]]:
    native_df = load_unsw_filtered_native(
        domain_row=domain_row,
        unsw_native_cols=unsw_native_cols,
        max_rows=max_rows,
        seed=seed,
    )
    return align_single_domain(native_df, dataset="unsw", alignment_df=alignment_df)


def load_domain_pool(
    domain_ids: list[str],
    registry: pd.DataFrame,
    alignment_df: pd.DataFrame,
    iot_native_cols: list[str],
    unsw_native_cols: list[str],
    scenario_dir: Path,
    total_cap: int | None,
    seed: int,
    cache: dict[tuple[str, int | None, int], pd.DataFrame],
) -> tuple[pd.DataFrame, list[str]]:
    domain_rows = registry[registry["domain_id"].isin(domain_ids)].copy()
    if domain_rows.empty:
        raise ValueError(f"No domain rows found for ids: {domain_ids}")

    datasets = domain_rows["dataset"].unique().tolist()
    if len(datasets) != 1:
        raise ValueError(f"Mixed datasets in one domain pool are not supported: {datasets}")

    caps_by_domain = allocate_caps(domain_rows, total_cap=total_cap)
    aligned_parts: list[pd.DataFrame] = []
    feature_cols: list[str] | None = None

    for _, domain_row in domain_rows.iterrows():
        cache_key = (str(domain_row["domain_id"]), caps_by_domain[domain_row["domain_id"]], seed)
        if cache_key in cache:
            aligned_df = cache[cache_key]
        else:
            if domain_row["dataset"] == "iot23":
                aligned_df, feature_cols_local = load_iot_domain(
                    domain_row=domain_row,
                    iot_native_cols=iot_native_cols,
                    alignment_df=alignment_df,
                    scenario_dir=scenario_dir,
                    max_rows=caps_by_domain[domain_row["domain_id"]],
                    seed=seed,
                )
            else:
                aligned_df, feature_cols_local = load_unsw_domain(
                    domain_row=domain_row,
                    unsw_native_cols=unsw_native_cols,
                    alignment_df=alignment_df,
                    max_rows=caps_by_domain[domain_row["domain_id"]],
                    seed=seed,
                )
            cache[cache_key] = aligned_df
            if feature_cols is None:
                feature_cols = feature_cols_local

        aligned_parts.append(aligned_df)
        if feature_cols is None:
            feature_cols = [col for col in aligned_df.columns if col != "label"]

    combined = pd.concat(aligned_parts, ignore_index=True)
    combined = base.maybe_sample_rows(combined, max_rows=total_cap, random_state=seed)
    return combined.reset_index(drop=True), (feature_cols or [])


def resolve_target_train_domain_ids(target_row: pd.Series, registry: pd.DataFrame) -> tuple[list[str], str]:
    exact = registry[
        (registry["dataset"] == target_row["dataset"])
        & (registry["split"] == "train")
        & (registry["grouping"] == target_row["grouping"])
        & (registry["group_value"] == target_row["group_value"])
    ]
    if not exact.empty:
        return exact["domain_id"].tolist(), "exact_group_match"

    family = registry[
        (registry["dataset"] == target_row["dataset"])
        & (registry["split"] == "train")
        & (registry["domain_family"] == target_row["domain_family"])
    ]
    if not family.empty:
        return family["domain_id"].tolist(), "family_train_pool"

    full_train = registry[
        (registry["dataset"] == target_row["dataset"])
        & (registry["split"] == "train")
    ]
    if not full_train.empty:
        return full_train["domain_id"].tolist(), "full_train_pool"

    raise ValueError(f"Could not resolve target-train pool for domain {target_row['domain_id']}")


def attach_pair_metadata(
    result: dict[str, object],
    pair_row: pd.Series,
    source_train_ids: list[str],
    target_train_ids: list[str],
    target_resolution: str,
    seed: int,
) -> dict[str, object]:
    return {
        "pair_id": pair_row["pair_id"],
        "pair_family": pair_row["pair_family"],
        "source_domain_id": pair_row["source_domain_id"],
        "target_test_domain_id": pair_row["target_domain_id"],
        "resolved_source_train_ids": "|".join(source_train_ids),
        "resolved_target_train_ids": "|".join(target_train_ids),
        "target_train_resolution": target_resolution,
        "seed": seed,
        **result,
    }


def has_both_classes(df: pd.DataFrame) -> bool:
    if "label" not in df.columns or df.empty:
        return False
    return df["label"].nunique(dropna=True) >= 2


def dataset_class_counts(df: pd.DataFrame) -> dict[str, int]:
    if "label" not in df.columns or df.empty:
        return {"n_benign": 0, "n_attack": 0}

    counts = df["label"].value_counts().to_dict()
    return {
        "n_benign": int(counts.get(0, 0)),
        "n_attack": int(counts.get(1, 0)),
    }


def main() -> None:
    args = parse_args()

    registry = load_registry(args.domain_registry)
    pairs = load_pairs(args.domain_pairs)
    if args.pair_family:
        pairs = pairs[pairs["pair_family"] == args.pair_family].copy()
    if args.pair_limit is not None:
        pairs = pairs.head(args.pair_limit).copy()
    pairs = pairs.reset_index(drop=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    alignment_df = base.load_alignment_table(
        Path(args.alignment_csv),
        include_review_features=args.include_review_features,
    )
    iot_native_cols, unsw_native_cols, aligned_features = aligned_feature_lists(alignment_df)
    scenario_dir = Path(args.iot_scenario_dir)

    with open(out_dir / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "domain_registry": args.domain_registry,
                "domain_pairs": args.domain_pairs,
                "pair_count": int(len(pairs)),
                "seeds": args.seeds,
                "target_fractions": args.target_fractions,
                "source_max_rows": args.source_max_rows,
                "target_train_max_rows": args.target_train_max_rows,
                "target_test_max_rows": args.target_test_max_rows,
                "balance_target_train": args.balance_target_train,
                "target_balance_ratio": args.target_balance_ratio,
                "calibration_fraction": args.calibration_fraction,
                "thresholds": args.thresholds,
                "pretrain_estimators": args.pretrain_estimators,
                "adapt_estimators": args.adapt_estimators,
                "target_only_estimators": args.target_only_estimators,
                "xgb_max_depth": args.xgb_max_depth,
                "xgb_learning_rate": args.xgb_learning_rate,
                "alignment_csv": args.alignment_csv,
                "n_aligned_features": len(aligned_features),
                "aligned_features": aligned_features,
            },
            handle,
            indent=2,
        )

    cache: dict[tuple[str, int | None, int], pd.DataFrame] = {}
    summary_rows: list[dict[str, object]] = []

    registry_by_id = registry.set_index("domain_id", drop=False)

    for pair_idx, (_, pair_row) in enumerate(pairs.iterrows(), start=1):
        source_row = registry_by_id.loc[pair_row["source_domain_id"]]
        target_test_row = registry_by_id.loc[pair_row["target_domain_id"]]
        target_train_ids, target_resolution = resolve_target_train_domain_ids(target_test_row, registry)
        source_train_ids = [str(pair_row["source_domain_id"])]

        pair_dir = out_dir / f"pair_{pair_idx:03d}__{slugify(pair_row['pair_id'], max_len=100)}"
        pair_dir.mkdir(parents=True, exist_ok=True)

        for seed in args.seeds:
            seeded_args = make_args_for_seed(args, seed)
            seed_dir = pair_dir / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)

            source_train_df, feature_cols = load_domain_pool(
                domain_ids=source_train_ids,
                registry=registry,
                alignment_df=alignment_df,
                iot_native_cols=iot_native_cols,
                unsw_native_cols=unsw_native_cols,
                scenario_dir=scenario_dir,
                total_cap=args.source_max_rows,
                seed=seed,
                cache=cache,
            )
            target_train_full_df, _ = load_domain_pool(
                domain_ids=target_train_ids,
                registry=registry,
                alignment_df=alignment_df,
                iot_native_cols=iot_native_cols,
                unsw_native_cols=unsw_native_cols,
                scenario_dir=scenario_dir,
                total_cap=args.target_train_max_rows,
                seed=seed,
                cache=cache,
            )
            target_test_df, _ = load_domain_pool(
                domain_ids=[str(pair_row["target_domain_id"])],
                registry=registry,
                alignment_df=alignment_df,
                iot_native_cols=iot_native_cols,
                unsw_native_cols=unsw_native_cols,
                scenario_dir=scenario_dir,
                total_cap=args.target_test_max_rows,
                seed=seed,
                cache=cache,
            )

            source_counts = dataset_class_counts(source_train_df)
            target_train_counts = dataset_class_counts(target_train_full_df)
            target_test_counts = dataset_class_counts(target_test_df)

            if not has_both_classes(source_train_df):
                summary_rows.append(
                    {
                        "pair_id": pair_row["pair_id"],
                        "pair_family": pair_row["pair_family"],
                        "source_domain_id": pair_row["source_domain_id"],
                        "target_test_domain_id": pair_row["target_domain_id"],
                        "resolved_source_train_ids": "|".join(source_train_ids),
                        "resolved_target_train_ids": "|".join(target_train_ids),
                        "target_train_resolution": target_resolution,
                        "seed": seed,
                        "direction": str(pair_row["pair_id"]),
                        "condition": "skipped_invalid_source",
                        "target_fraction": math.nan,
                        "n_source_train": int(len(source_train_df)),
                        "n_target_fit": int(len(target_train_full_df)),
                        "n_target_calibration": 0,
                        "n_target_test": int(len(target_test_df)),
                        "source_n_benign": source_counts["n_benign"],
                        "source_n_attack": source_counts["n_attack"],
                        "target_train_n_benign": target_train_counts["n_benign"],
                        "target_train_n_attack": target_train_counts["n_attack"],
                        "target_test_n_benign": target_test_counts["n_benign"],
                        "target_test_n_attack": target_test_counts["n_attack"],
                        "skip_reason": "source_train_single_class",
                    }
                )
                continue

            if not has_both_classes(target_train_full_df):
                summary_rows.append(
                    {
                        "pair_id": pair_row["pair_id"],
                        "pair_family": pair_row["pair_family"],
                        "source_domain_id": pair_row["source_domain_id"],
                        "target_test_domain_id": pair_row["target_domain_id"],
                        "resolved_source_train_ids": "|".join(source_train_ids),
                        "resolved_target_train_ids": "|".join(target_train_ids),
                        "target_train_resolution": target_resolution,
                        "seed": seed,
                        "direction": str(pair_row["pair_id"]),
                        "condition": "skipped_invalid_target_train",
                        "target_fraction": math.nan,
                        "n_source_train": int(len(source_train_df)),
                        "n_target_fit": int(len(target_train_full_df)),
                        "n_target_calibration": 0,
                        "n_target_test": int(len(target_test_df)),
                        "source_n_benign": source_counts["n_benign"],
                        "source_n_attack": source_counts["n_attack"],
                        "target_train_n_benign": target_train_counts["n_benign"],
                        "target_train_n_attack": target_train_counts["n_attack"],
                        "target_test_n_benign": target_test_counts["n_benign"],
                        "target_test_n_attack": target_test_counts["n_attack"],
                        "skip_reason": "target_train_single_class",
                    }
                )
                continue

            if not has_both_classes(target_test_df):
                summary_rows.append(
                    {
                        "pair_id": pair_row["pair_id"],
                        "pair_family": pair_row["pair_family"],
                        "source_domain_id": pair_row["source_domain_id"],
                        "target_test_domain_id": pair_row["target_domain_id"],
                        "resolved_source_train_ids": "|".join(source_train_ids),
                        "resolved_target_train_ids": "|".join(target_train_ids),
                        "target_train_resolution": target_resolution,
                        "seed": seed,
                        "direction": str(pair_row["pair_id"]),
                        "condition": "skipped_invalid_target_test",
                        "target_fraction": math.nan,
                        "n_source_train": int(len(source_train_df)),
                        "n_target_fit": int(len(target_train_full_df)),
                        "n_target_calibration": 0,
                        "n_target_test": int(len(target_test_df)),
                        "source_n_benign": source_counts["n_benign"],
                        "source_n_attack": source_counts["n_attack"],
                        "target_train_n_benign": target_train_counts["n_benign"],
                        "target_train_n_attack": target_train_counts["n_attack"],
                        "target_test_n_benign": target_test_counts["n_benign"],
                        "target_test_n_attack": target_test_counts["n_attack"],
                        "skip_reason": "target_test_single_class",
                    }
                )
                continue

            source_only_result = base.evaluate_source_only(
                source_train_df=source_train_df,
                target_test_df=target_test_df,
                feature_cols=feature_cols,
                source_name=str(pair_row["source_domain_id"]),
                target_name=str(pair_row["target_domain_id"]),
                direction_dir=seed_dir,
                args=seeded_args,
            )
            summary_rows.append(
                attach_pair_metadata(
                    result=source_only_result,
                    pair_row=pair_row,
                    source_train_ids=source_train_ids,
                    target_train_ids=target_train_ids,
                    target_resolution=target_resolution,
                    seed=seed,
                )
            )

            for fraction in args.target_fractions:
                sampled_target_train = base.sample_target_fraction(
                    df=target_train_full_df,
                    fraction=fraction,
                    random_state=seed,
                )
                if args.balance_target_train:
                    sampled_target_train = base.balance_binary_target_train(
                        df=sampled_target_train,
                        random_state=seed,
                        majority_to_minority_ratio=args.target_balance_ratio,
                    )

                target_fit_df, target_calib_df, _ = updated.split_target_train_for_calibration(
                    df=sampled_target_train,
                    calibration_fraction=args.calibration_fraction,
                    random_state=seed,
                )

                sampled_counts = dataset_class_counts(sampled_target_train)
                if not has_both_classes(sampled_target_train):
                    summary_rows.append(
                        {
                            "pair_id": pair_row["pair_id"],
                            "pair_family": pair_row["pair_family"],
                            "source_domain_id": pair_row["source_domain_id"],
                            "target_test_domain_id": pair_row["target_domain_id"],
                            "resolved_source_train_ids": "|".join(source_train_ids),
                            "resolved_target_train_ids": "|".join(target_train_ids),
                            "target_train_resolution": target_resolution,
                            "seed": seed,
                            "direction": str(pair_row["pair_id"]),
                            "condition": "skipped_invalid_sampled_target",
                            "target_fraction": fraction,
                            "n_source_train": int(len(source_train_df)),
                            "n_target_fit": int(len(sampled_target_train)),
                            "n_target_calibration": 0,
                            "n_target_test": int(len(target_test_df)),
                            "source_n_benign": source_counts["n_benign"],
                            "source_n_attack": source_counts["n_attack"],
                            "target_train_n_benign": sampled_counts["n_benign"],
                            "target_train_n_attack": sampled_counts["n_attack"],
                            "target_test_n_benign": target_test_counts["n_benign"],
                            "target_test_n_attack": target_test_counts["n_attack"],
                            "skip_reason": "sampled_target_single_class",
                        }
                    )
                    continue

                target_only_result = updated.evaluate_target_only_updated(
                    target_fit_df=target_fit_df,
                    target_calib_df=target_calib_df,
                    target_test_df=target_test_df,
                    feature_cols=feature_cols,
                    direction_label=str(pair_row["pair_id"]),
                    fraction=fraction,
                    direction_dir=seed_dir,
                    args=seeded_args,
                )
                summary_rows.append(
                    attach_pair_metadata(
                        result=target_only_result,
                        pair_row=pair_row,
                        source_train_ids=source_train_ids,
                        target_train_ids=target_train_ids,
                        target_resolution=target_resolution,
                        seed=seed,
                    )
                )

                transfer_result = updated.evaluate_transfer_learning_updated(
                    source_train_df=source_train_df,
                    target_fit_df=target_fit_df,
                    target_calib_df=target_calib_df,
                    target_test_df=target_test_df,
                    feature_cols=feature_cols,
                    direction_label=str(pair_row["pair_id"]),
                    fraction=fraction,
                    direction_dir=seed_dir,
                    args=seeded_args,
                )
                summary_rows.append(
                    attach_pair_metadata(
                        result=transfer_result,
                        pair_row=pair_row,
                        source_train_ids=source_train_ids,
                        target_train_ids=target_train_ids,
                        target_resolution=target_resolution,
                        seed=seed,
                    )
                )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "pairwise_hypothesis_summary.csv", index=False)

    print("Pairwise transfer-learning hypothesis runs complete.")
    print(summary_df.head(20).to_string(index=False))
    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
