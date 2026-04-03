from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

try:
    from data_prep_iot23 import get_split
except ModuleNotFoundError:
    from data_preprocessing.data_prep_iot23 import get_split


def save_json(obj: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def normalize_table_schema(table: pa.Table) -> pa.Table:
    target_types = {
        "ts": pa.float64(),
        "scenario": pa.string(),
        "split": pa.string(),
        "proto": pa.string(),
        "service": pa.string(),
        "conn_state": pa.string(),
        "duration": pa.float64(),
        "orig_bytes": pa.float64(),
        "resp_bytes": pa.float64(),
        "missed_bytes": pa.float64(),
        "orig_pkts": pa.float64(),
        "orig_ip_bytes": pa.float64(),
        "resp_pkts": pa.float64(),
        "resp_ip_bytes": pa.float64(),
        "bytes_ratio": pa.float64(),
        "pkts_ratio": pa.float64(),
        "orig_bytes_per_pkt": pa.float64(),
        "resp_bytes_per_pkt": pa.float64(),
        "label": pa.string(),
        "detailed_label": pa.string(),
        "label_binary": pa.int64(),
        "label_phase": pa.string(),
    }

    cols = []
    names = []

    for name in table.column_names:
        col = table[name]
        if name in target_types:
            col = pc.cast(col, target_types[name], safe=False)
        cols.append(col)
        names.append(name)

    return pa.Table.from_arrays(cols, names=names)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Memory-safe export of train/val/test splits from full IoT-23 per-scenario parquet files"
    )
    parser.add_argument(
        "--scenario_dir",
        required=True,
        help="Directory containing per-scenario parquet files",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output directory (usually Datasets/IoT23/processed_full/iot23)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100_000,
        help="Parquet batch size for streaming",
    )
    args = parser.parse_args()

    scenario_dir = Path(args.scenario_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenario_files = sorted(scenario_dir.glob("*.parquet"))
    if not scenario_files:
        raise FileNotFoundError(f"No scenario parquet files found in {scenario_dir}")

    train_path = out_dir / "train.parquet"
    val_path = out_dir / "val.parquet"
    test_path = out_dir / "test.parquet"

    # Remove partial old outputs if they exist
    for path in [train_path, val_path, test_path]:
        if path.exists():
            path.unlink()

    train_writer = None
    val_writer = None
    test_writer = None

    scenario_summary_rows = []
    split_counts = {
        "train": {"rows": 0, "benign": 0, "malicious": 0, "scenarios": []},
        "val": {"rows": 0, "benign": 0, "malicious": 0, "scenarios": []},
        "test": {"rows": 0, "benign": 0, "malicious": 0, "scenarios": []},
    }

    try:
        for p in scenario_files:
            scenario_name = p.stem
            split = get_split(scenario_name)
            print(f"[INFO] Streaming {scenario_name} -> {split}")

            pf = pq.ParquetFile(p)

            scenario_rows = 0
            scenario_benign = 0
            scenario_malicious = 0

            for batch in pf.iter_batches(batch_size=args.batch_size):
                table = pa.Table.from_batches([batch])

                n = table.num_rows
                scenario_arr = pa.array([scenario_name] * n, type=pa.string())
                split_arr = pa.array([split] * n, type=pa.string())

                # overwrite/add scenario column
                if "scenario" in table.column_names:
                    idx = table.column_names.index("scenario")
                    table = table.set_column(idx, "scenario", scenario_arr)
                else:
                    table = table.append_column("scenario", scenario_arr)

                # overwrite/add split column
                if "split" in table.column_names:
                    idx = table.column_names.index("split")
                    table = table.set_column(idx, "split", split_arr)
                else:
                    table = table.append_column("split", split_arr)

                table = normalize_table_schema(table)

                if "label_binary" not in table.column_names:
                    raise ValueError(f"{scenario_name} is missing label_binary")

                label_col = table["label_binary"]
                benign = pc.sum(pc.equal(label_col, 0)).as_py() or 0
                malicious = pc.sum(pc.equal(label_col, 1)).as_py() or 0

                scenario_rows += n
                scenario_benign += int(benign)
                scenario_malicious += int(malicious)

                if split == "train":
                    if train_writer is None:
                        train_writer = pq.ParquetWriter(train_path, table.schema)
                    train_writer.write_table(table)
                elif split == "val":
                    if val_writer is None:
                        val_writer = pq.ParquetWriter(val_path, table.schema)
                    val_writer.write_table(table)
                elif split == "test":
                    if test_writer is None:
                        test_writer = pq.ParquetWriter(test_path, table.schema)
                    test_writer.write_table(table)
                else:
                    raise ValueError(f"Unknown split: {split}")

                del table
                gc.collect()

            split_counts[split]["rows"] += scenario_rows
            split_counts[split]["benign"] += scenario_benign
            split_counts[split]["malicious"] += scenario_malicious
            split_counts[split]["scenarios"].append(scenario_name)

            scenario_summary_rows.append(
                {
                    "scenario": scenario_name,
                    "split": split,
                    "n_rows": scenario_rows,
                    "n_benign": scenario_benign,
                    "n_malicious": scenario_malicious,
                }
            )

    finally:
        if train_writer is not None:
            train_writer.close()
        if val_writer is not None:
            val_writer.close()
        if test_writer is not None:
            test_writer.close()

    scenario_summary = pd.DataFrame(scenario_summary_rows).sort_values(["split", "scenario"])
    scenario_summary.to_csv(out_dir / "scenario_summary.csv", index=False)

    split_summary = {
        split: {
            "n_rows": int(info["rows"]),
            "n_scenarios": int(len(info["scenarios"])),
            "scenarios": sorted(info["scenarios"]),
            "label_binary_counts": {
                "0": int(info["benign"]),
                "1": int(info["malicious"]),
            },
        }
        for split, info in split_counts.items()
    }
    save_json(split_summary, out_dir / "split_summary.json")

    print("\n[INFO] Saved:")
    print(f"  {train_path}")
    print(f"  {val_path}")
    print(f"  {test_path}")
    print(f"  {out_dir / 'scenario_summary.csv'}")
    print(f"  {out_dir / 'split_summary.json'}")


if __name__ == "__main__":
    main()