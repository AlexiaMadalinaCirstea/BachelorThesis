from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create many lightweight pseudo-domain definitions and cross-dataset domain pairs "
            "for transfer-learning hypothesis testing. "
            "This script writes metadata only; it does not materialize heavy subset files."
        )
    )
    parser.add_argument(
        "--iot_scenario_summary",
        default="Datasets/IoT23/processed_full/iot23/scenario_summary.csv",
        help="IoT-23 scenario summary CSV with per-scenario row counts.",
    )
    parser.add_argument(
        "--iot_train",
        default="Datasets/IoT23/processed_full/iot23/train.parquet",
        help="IoT-23 train parquet path referenced in the domain registry.",
    )
    parser.add_argument(
        "--iot_val",
        default="Datasets/IoT23/processed_full/iot23/val.parquet",
        help="IoT-23 val parquet path referenced in the domain registry.",
    )
    parser.add_argument(
        "--iot_test",
        default="Datasets/IoT23/processed_full/iot23/test.parquet",
        help="IoT-23 test parquet path referenced in the domain registry.",
    )
    parser.add_argument(
        "--unsw_train",
        default="Datasets/UNSW-NB15/UNSW-NB15 dataset/CSV Files/Training and Testing Sets/UNSW_NB15_training-set.csv",
        help="UNSW-NB15 training CSV used for chunked metadata aggregation.",
    )
    parser.add_argument(
        "--unsw_test",
        default="Datasets/UNSW-NB15/UNSW-NB15 dataset/CSV Files/Training and Testing Sets/UNSW_NB15_testing-set.csv",
        help="UNSW-NB15 testing CSV used for chunked metadata aggregation.",
    )
    parser.add_argument(
        "--unsw_group_columns",
        nargs="+",
        default=["attack_cat", "proto"],
        help=(
            "UNSW columns used to define pseudo-domains. "
            "Keep this small at first to avoid exploding the number of pairs."
        ),
    )
    parser.add_argument(
        "--unsw_chunksize",
        type=int,
        default=250000,
        help="Chunk size for UNSW metadata aggregation.",
    )
    parser.add_argument(
        "--min_rows_per_domain",
        type=int,
        default=5000,
        help="Minimum number of rows required for a domain to be kept.",
    )
    parser.add_argument(
        "--min_attack_rows",
        type=int,
        default=100,
        help="Minimum number of attack rows required for a domain to be kept.",
    )
    parser.add_argument(
        "--min_benign_rows",
        type=int,
        default=100,
        help="Minimum number of benign rows required for a domain to be kept.",
    )
    parser.add_argument(
        "--top_k_unsw_per_grouping",
        type=int,
        default=8,
        help="Keep only the largest K UNSW domains per grouping and split after filtering.",
    )
    parser.add_argument(
        "--train_split_only",
        action="store_true",
        help="Keep only domains defined on train splits.",
    )
    parser.add_argument(
        "--top_iot_benign_rich",
        type=int,
        default=5,
        help="Number of IoT train scenarios used in the benign-rich grouped source domain.",
    )
    parser.add_argument(
        "--out_dir",
        default="transfer_learning/hypothesis/generated_pairs",
        help="Output directory for domain and pair manifests.",
    )
    return parser.parse_args()


def slugify(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "missing"


def normalize_binary_labels(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).astype("int8")

    normalized = series.astype("string").str.strip().str.lower()
    mapping = {
        "0": 0,
        "1": 1,
        "normal": 0,
        "benign": 0,
        "background": 0,
        "attack": 1,
        "anomaly": 1,
        "malicious": 1,
    }
    mapped = normalized.map(mapping)
    return mapped.fillna(0).astype("int8")


def iot_split_path_map(args: argparse.Namespace) -> dict[str, str]:
    return {
        "train": args.iot_train,
        "val": args.iot_val,
        "test": args.iot_test,
    }


def classify_iot_scenario_family(scenario_name: str) -> str:
    lowered = scenario_name.lower()
    if "honeypot" in lowered:
        return "honeypot_benign"
    if "malware" in lowered:
        return "iot_malware"
    return "other"


def build_iot_domains(args: argparse.Namespace) -> pd.DataFrame:
    summary = pd.read_csv(args.iot_scenario_summary)
    split_to_path = iot_split_path_map(args)

    required_cols = {"scenario", "split", "n_rows", "n_benign", "n_malicious"}
    missing = required_cols - set(summary.columns)
    if missing:
        raise ValueError(f"IoT scenario summary is missing required columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    for record in summary.to_dict(orient="records"):
        split = str(record["split"]).strip().lower()
        scenario = str(record["scenario"]).strip()
        n_rows = int(record["n_rows"])
        n_attack = int(record["n_malicious"])
        n_benign = int(record["n_benign"])

        if (
            n_rows < args.min_rows_per_domain
            or n_attack < args.min_attack_rows
            or n_benign < args.min_benign_rows
        ):
            continue
        if args.train_split_only and split != "train":
            continue
        if split not in split_to_path:
            continue

        scenario_slug = slugify(scenario)
        family = classify_iot_scenario_family(scenario)
        rows.append(
            {
                "domain_id": f"iot23__scenario__{split}__{scenario_slug}",
                "dataset": "iot23",
                "grouping": "scenario",
                "group_value": scenario,
                "group_slug": scenario_slug,
                "domain_family": family,
                "split": split,
                "source_path": split_to_path[split],
                "n_rows": n_rows,
                "n_benign": n_benign,
                "n_attack": n_attack,
                "attack_rate": n_attack / n_rows if n_rows else math.nan,
                "filter_json": json.dumps(
                    {
                        "dataset": "iot23",
                        "path": split_to_path[split],
                        "filters": [
                            {
                                "column": "scenario",
                                "op": "eq",
                                "value": scenario,
                            }
                        ],
                    },
                    sort_keys=True,
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(["split", "n_rows"], ascending=[True, False]).reset_index(drop=True)


def build_iot_source_groups(iot_domains: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    train_domains = iot_domains[iot_domains["split"] == "train"].copy()
    if train_domains.empty:
        return pd.DataFrame(columns=iot_domains.columns)

    grouped_specs: list[tuple[str, pd.DataFrame]] = [("all_train_mixed", train_domains)]

    moderate = train_domains[train_domains["attack_rate"] <= 0.98].copy()
    if not moderate.empty:
        grouped_specs.append(("moderate_attack_mix", moderate))

    benign_rich = train_domains.sort_values("n_benign", ascending=False).head(args.top_iot_benign_rich).copy()
    if not benign_rich.empty:
        grouped_specs.append(("benign_rich_mix", benign_rich))

    records: list[dict[str, object]] = []
    seen_names: set[str] = set()
    for group_name, group_df in grouped_specs:
        if group_name in seen_names or group_df.empty:
            continue
        seen_names.add(group_name)

        scenarios = group_df["group_value"].tolist()
        n_rows = int(group_df["n_rows"].sum())
        n_benign = int(group_df["n_benign"].sum())
        n_attack = int(group_df["n_attack"].sum())

        records.append(
            {
                "domain_id": f"iot23__source_group__train__{slugify(group_name)}",
                "dataset": "iot23",
                "grouping": "source_group",
                "group_value": group_name,
                "group_slug": slugify(group_name),
                "domain_family": "iot_source_group",
                "split": "train",
                "source_path": args.iot_train,
                "n_rows": n_rows,
                "n_benign": n_benign,
                "n_attack": n_attack,
                "attack_rate": n_attack / n_rows if n_rows else math.nan,
                "filter_json": json.dumps(
                    {
                        "dataset": "iot23",
                        "path": args.iot_train,
                        "filters": [
                            {
                                "column": "scenario",
                                "op": "in",
                                "values": scenarios,
                            }
                        ],
                    },
                    sort_keys=True,
                ),
            }
        )

    if not records:
        return pd.DataFrame(columns=iot_domains.columns)

    return pd.DataFrame(records).sort_values("n_rows", ascending=False).reset_index(drop=True)


def aggregate_unsw_group_counts(
    csv_path: str,
    split_name: str,
    group_columns: list[str],
    chunksize: int,
) -> pd.DataFrame:
    usecols = sorted(set(group_columns + ["label"]))
    aggregated_chunks: list[pd.DataFrame] = []

    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize):
        chunk = chunk.copy()
        chunk["label"] = normalize_binary_labels(chunk["label"])

        per_group_frames = []
        for grouping in group_columns:
            local = chunk[[grouping, "label"]].copy()
            local[grouping] = local[grouping].astype("string").fillna("missing").str.strip()
            local = local[local[grouping].ne("")]

            grouped = local.groupby(grouping, dropna=False)["label"].agg(["size", "sum"]).reset_index()
            grouped = grouped.rename(
                columns={
                    grouping: "group_value",
                    "size": "n_rows",
                    "sum": "n_attack",
                }
            )
            grouped["grouping"] = grouping
            grouped["split"] = split_name
            per_group_frames.append(grouped)

        if per_group_frames:
            aggregated_chunks.append(pd.concat(per_group_frames, ignore_index=True))

    if not aggregated_chunks:
        return pd.DataFrame(columns=["grouping", "group_value", "split", "n_rows", "n_attack", "n_benign"])

    aggregated = pd.concat(aggregated_chunks, ignore_index=True)
    aggregated = (
        aggregated.groupby(["grouping", "group_value", "split"], as_index=False)[["n_rows", "n_attack"]]
        .sum()
        .sort_values(["grouping", "split", "n_rows"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    aggregated["n_benign"] = aggregated["n_rows"] - aggregated["n_attack"]
    return aggregated


def build_unsw_domains(args: argparse.Namespace) -> pd.DataFrame:
    aggregated = pd.concat(
        [
            aggregate_unsw_group_counts(
                csv_path=args.unsw_train,
                split_name="train",
                group_columns=args.unsw_group_columns,
                chunksize=args.unsw_chunksize,
            ),
            aggregate_unsw_group_counts(
                csv_path=args.unsw_test,
                split_name="test",
                group_columns=args.unsw_group_columns,
                chunksize=args.unsw_chunksize,
            ),
        ],
        ignore_index=True,
    )

    if aggregated.empty:
        return aggregated

    filtered = aggregated[
        (aggregated["n_rows"] >= args.min_rows_per_domain)
        & (aggregated["n_attack"] >= args.min_attack_rows)
        & (aggregated["n_benign"] >= args.min_benign_rows)
    ].copy()

    if args.train_split_only:
        filtered = filtered[filtered["split"] == "train"].copy()

    # Keep only the most substantial UNSW domains per grouping and split to avoid pair explosion.
    filtered["rank_within_grouping"] = filtered.groupby(["grouping", "split"])["n_rows"].rank(
        method="first",
        ascending=False,
    )
    filtered = filtered[filtered["rank_within_grouping"] <= args.top_k_unsw_per_grouping].copy()

    path_map = {
        "train": args.unsw_train,
        "test": args.unsw_test,
    }

    records: list[dict[str, object]] = []
    for row in filtered.to_dict(orient="records"):
        grouping = str(row["grouping"])
        group_value = str(row["group_value"])
        split = str(row["split"])
        group_slug = slugify(group_value)
        source_path = path_map[split]

        records.append(
            {
                "domain_id": f"unsw__{slugify(grouping)}__{split}__{group_slug}",
                "dataset": "unsw",
                "grouping": grouping,
                "group_value": group_value,
                "group_slug": group_slug,
                "domain_family": grouping,
                "split": split,
                "source_path": source_path,
                "n_rows": int(row["n_rows"]),
                "n_benign": int(row["n_benign"]),
                "n_attack": int(row["n_attack"]),
                "attack_rate": float(row["n_attack"] / row["n_rows"]) if row["n_rows"] else math.nan,
                "filter_json": json.dumps(
                    {
                        "dataset": "unsw",
                        "path": source_path,
                        "filters": [
                            {
                                "column": grouping,
                                "op": "eq",
                                "value": group_value,
                            }
                        ],
                    },
                    sort_keys=True,
                ),
            }
        )

    return pd.DataFrame(records).sort_values(["grouping", "split", "n_rows"], ascending=[True, True, False]).reset_index(
        drop=True
    )


def build_cross_dataset_pairs(domains: pd.DataFrame) -> pd.DataFrame:
    if domains.empty:
        return pd.DataFrame()

    iot_domains = domains[domains["dataset"] == "iot23"].copy()
    unsw_domains = domains[domains["dataset"] == "unsw"].copy()

    pair_rows: list[dict[str, object]] = []
    source_iot = iot_domains[
        (iot_domains["split"] == "train") & (iot_domains["grouping"] == "source_group")
    ].copy()
    if source_iot.empty:
        source_iot = iot_domains[
            (iot_domains["split"] == "train") & (iot_domains["grouping"] == "scenario")
        ].copy()
    source_unsw = unsw_domains[unsw_domains["split"] == "train"].copy()
    target_iot = iot_domains[
        (iot_domains["split"].isin(["val", "test"])) & (iot_domains["grouping"] == "scenario")
    ].copy()
    target_unsw = unsw_domains[unsw_domains["split"] == "test"].copy()

    for source_df, target_df, pair_family in [
        (source_iot, target_unsw, "iot23_to_unsw"),
        (source_unsw, target_iot, "unsw_to_iot23"),
    ]:
        for source in source_df.to_dict(orient="records"):
            for target in target_df.to_dict(orient="records"):
                pair_rows.append(
                    {
                        "pair_id": (
                            f"{source['domain_id']}__TO__{target['domain_id']}"
                        ),
                        "pair_family": pair_family,
                        "source_domain_id": source["domain_id"],
                        "target_domain_id": target["domain_id"],
                        "source_dataset": source["dataset"],
                        "target_dataset": target["dataset"],
                        "source_grouping": source["grouping"],
                        "target_grouping": target["grouping"],
                        "source_domain_family": source["domain_family"],
                        "target_domain_family": target["domain_family"],
                        "source_split": source["split"],
                        "target_split": target["split"],
                        "source_n_rows": source["n_rows"],
                        "target_n_rows": target["n_rows"],
                        "source_attack_rate": source["attack_rate"],
                        "target_attack_rate": target["attack_rate"],
                    }
                )

    return pd.DataFrame(pair_rows).sort_values(
        ["pair_family", "source_n_rows", "target_n_rows"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    iot_domains_base = build_iot_domains(args)
    iot_source_groups = build_iot_source_groups(iot_domains_base, args)
    iot_domains = pd.concat([iot_domains_base, iot_source_groups], ignore_index=True)
    unsw_domains = build_unsw_domains(args)

    domains = pd.concat([iot_domains, unsw_domains], ignore_index=True)
    if domains.empty:
        raise ValueError("No domains survived filtering. Relax the thresholds and rerun.")

    domain_pairs = build_cross_dataset_pairs(domains)

    domains.to_csv(out_dir / "domain_registry.csv", index=False)
    domain_pairs.to_csv(out_dir / "domain_pairs.csv", index=False)

    summary = {
        "n_domains_total": int(len(domains)),
        "n_iot_domains": int(len(iot_domains)),
        "n_unsw_domains": int(len(unsw_domains)),
        "n_pairs_total": int(len(domain_pairs)),
        "iot_groupings": sorted(iot_domains["grouping"].dropna().unique().tolist()) if not iot_domains.empty else [],
        "unsw_groupings": sorted(unsw_domains["grouping"].dropna().unique().tolist()) if not unsw_domains.empty else [],
        "args": vars(args),
    }

    with open(out_dir / "domain_pair_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Saved domain registry to: {out_dir / 'domain_registry.csv'}")
    print(f"Saved domain pairs to: {out_dir / 'domain_pairs.csv'}")
    print(f"Saved summary to: {out_dir / 'domain_pair_summary.json'}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
