# THIS IS A UTILITY FILE SO I DO NOT DUPLICATE CODE IN THE BASELINES!

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

log = logging.getLogger(__name__)

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

CATEGORICAL_COLS = ["proto", "service", "conn_state"]
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS
META_COLS = ["scenario", "split", "label", "detailed_label", "label_phase", "ts"]
REQUIRED_BASE_COLS = ["scenario", "label_binary"]


@dataclass
class BinaryRunningMetrics:
    tn: int = 0
    fp: int = 0
    fn: int = 0
    tp: int = 0

    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        y_true = np.asarray(y_true, dtype=np.int8)
        y_pred = np.asarray(y_pred, dtype=np.int8)
        self.tp += int(((y_true == 1) & (y_pred == 1)).sum())
        self.tn += int(((y_true == 0) & (y_pred == 0)).sum())
        self.fp += int(((y_true == 0) & (y_pred == 1)).sum())
        self.fn += int(((y_true == 1) & (y_pred == 0)).sum())

    def as_dict(self) -> Dict[str, object]:
        support_0 = self.tn + self.fp
        support_1 = self.tp + self.fn
        total = support_0 + support_1

        p0 = _safe_div(self.tn, self.tn + self.fn)
        r0 = _safe_div(self.tn, self.tn + self.fp)
        f10 = _safe_f1(p0, r0)

        p1 = _safe_div(self.tp, self.tp + self.fp)
        r1 = _safe_div(self.tp, self.tp + self.fn)
        f11 = _safe_f1(p1, r1)

        precision_macro = float((p0 + p1) / 2.0)
        recall_macro = float((r0 + r1) / 2.0)
        f1_macro = float((f10 + f11) / 2.0)

        precision_weighted = float(_weighted_mean([p0, p1], [support_0, support_1]))
        recall_weighted = float(_weighted_mean([r0, r1], [support_0, support_1]))
        f1_weighted = float(_weighted_mean([f10, f11], [support_0, support_1]))
        accuracy = float(_safe_div(self.tp + self.tn, total))

        return {
            "accuracy": accuracy,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "precision_weighted": precision_weighted,
            "recall_weighted": recall_weighted,
            "f1_weighted": f1_weighted,
            "confusion_matrix": {
                "labels": [0, 1],
                "matrix": [[self.tn, self.fp], [self.fn, self.tp]],
            },
            "per_class": {
                "0": {
                    "precision": float(p0),
                    "recall": float(r0),
                    "f1": float(f10),
                    "support": int(support_0),
                },
                "1": {
                    "precision": float(p1),
                    "recall": float(r1),
                    "f1": float(f11),
                    "support": int(support_1),
                },
            },
            "n_rows": int(total),
        }


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _safe_f1(precision: float, recall: float) -> float:
    return float((2.0 * precision * recall) / (precision + recall)) if (precision + recall) else 0.0


def _weighted_mean(values: list[float], weights: list[int]) -> float:
    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0
    return float(sum(v * w for v, w in zip(values, weights)) / total_weight)


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def validate_required_columns(schema_names: Iterable[str], target_col: str) -> None:
    required = set(FEATURE_COLS + REQUIRED_BASE_COLS + [target_col])
    missing = sorted(required - set(schema_names))
    if missing:
        raise AssertionError(f"Missing required columns: {missing}")


def get_dataset_and_schema(path: Path, target_col: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet path: {path}")
    dataset = ds.dataset(str(path), format="parquet")
    schema_names = dataset.schema.names
    validate_required_columns(schema_names, target_col)
    return dataset, dataset.schema


def iter_parquet_batches(
    path: Path,
    columns: list[str],
    batch_size: int,
    target_col: str,
) -> Iterator[pd.DataFrame]:
    dataset, _ = get_dataset_and_schema(path, target_col)
    scanner = dataset.scanner(columns=columns, batch_size=batch_size)
    for record_batch in scanner.to_batches():
        df = record_batch.to_pandas(types_mapper=pd.ArrowDtype)
        # convert to standard pandas dtypes we control
        for col in NUMERIC_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
        for col in CATEGORICAL_COLS:
            if col in df.columns:
                df[col] = df[col].astype("string").fillna("unknown")
        if target_col in df.columns:
            df[target_col] = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype("int8")
        if "scenario" in df.columns:
            df["scenario"] = df["scenario"].astype("string")
        yield df


def count_rows(path: Path) -> int:
    dataset = ds.dataset(str(path), format="parquet")
    return int(dataset.count_rows())


def collect_label_counts(path: Path, target_col: str, batch_size: int) -> Dict[int, int]:
    counts = {0: 0, 1: 0}
    for df in iter_parquet_batches(path=path, columns=[target_col], batch_size=batch_size, target_col=target_col):
        vc = df[target_col].value_counts(dropna=False)
        counts[0] += int(vc.get(0, 0))
        counts[1] += int(vc.get(1, 0))
    return counts


def sample_training_split(
    path: Path,
    target_col: str,
    batch_size: int,
    sample_frac: float,
    seed: int,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    if not (0 < sample_frac <= 1.0):
        raise ValueError("sample_frac must be in (0, 1]")

    rng = np.random.default_rng(seed)
    sampled_parts: list[pd.DataFrame] = []

    keep_by_class: Optional[Dict[int, int]] = None
    if max_rows is not None:
        counts = collect_label_counts(path=path, target_col=target_col, batch_size=batch_size)
        total = max(sum(counts.values()), 1)
        keep_by_class = {
            0: int(round(max_rows * (counts[0] / total))),
            1: int(round(max_rows * (counts[1] / total))),
        }
        keep_by_class[1] = min(keep_by_class[1], counts[1])
        keep_by_class[0] = min(max_rows - keep_by_class[1], counts[0])
        log.info("Target capped sample sizes by class: benign=%d attack=%d", keep_by_class[0], keep_by_class[1])

    kept_so_far = {0: 0, 1: 0}
    columns = FEATURE_COLS + ["scenario", target_col]
    for chunk_idx, df in enumerate(iter_parquet_batches(path=path, columns=columns, batch_size=batch_size, target_col=target_col), start=1):
        if sample_frac < 1.0:
            sampled_chunk_parts = []
            for label_value, group in df.groupby(target_col, sort=False):
                if group.empty:
                    continue
                take_n = int(np.ceil(len(group) * sample_frac))
                take_n = max(1, take_n) if len(group) > 0 else 0
                idx = rng.choice(len(group), size=min(take_n, len(group)), replace=False)
                sampled_chunk_parts.append(group.iloc[np.sort(idx)])
            chunk = pd.concat(sampled_chunk_parts, ignore_index=True) if sampled_chunk_parts else df.iloc[0:0].copy()
        else:
            chunk = df

        if keep_by_class is not None and not chunk.empty:
            capped_parts = []
            for label_value, group in chunk.groupby(target_col, sort=False):
                label_int = int(label_value)
                remaining = keep_by_class[label_int] - kept_so_far[label_int]
                if remaining <= 0:
                    continue
                if len(group) > remaining:
                    idx = rng.choice(len(group), size=remaining, replace=False)
                    group = group.iloc[np.sort(idx)]
                kept_so_far[label_int] += len(group)
                capped_parts.append(group)
            chunk = pd.concat(capped_parts, ignore_index=True) if capped_parts else chunk.iloc[0:0].copy()

        if not chunk.empty:
            sampled_parts.append(chunk)

        if chunk_idx % 25 == 0:
            log.info("Processed %d training chunks while sampling", chunk_idx)

        if keep_by_class is not None and kept_so_far[0] >= keep_by_class[0] and kept_so_far[1] >= keep_by_class[1]:
            break

    if not sampled_parts:
        raise AssertionError("Sampling produced an empty training set")

    sampled_df = pd.concat(sampled_parts, ignore_index=True)
    validate_required_columns(sampled_df.columns, target_col)
    return sampled_df


def prediction_schema(target_col: str) -> pa.Schema:
    return pa.schema([
        pa.field("scenario", pa.string()),
        pa.field(target_col, pa.int8()),
        pa.field("y_pred", pa.int8()),
        pa.field("y_score", pa.float32()),
    ])


def evaluate_split_in_batches(
    pipeline,
    split_path: Path,
    split_name: str,
    out_path: Path,
    target_col: str,
    batch_size: int,
    threshold: float = 0.5,
) -> Dict[str, object]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer: Optional[pq.ParquetWriter] = None
    running = BinaryRunningMetrics()

    columns = FEATURE_COLS + ["scenario", target_col]
    try:
        for batch_idx, df in enumerate(iter_parquet_batches(path=split_path, columns=columns, batch_size=batch_size, target_col=target_col), start=1):
            X = df[FEATURE_COLS]
            y_true = df[target_col].to_numpy(dtype=np.int8, copy=False)
            y_score = pipeline.predict_proba(X)[:, 1].astype(np.float32, copy=False)
            y_pred = (y_score >= threshold).astype(np.int8, copy=False)
            running.update(y_true=y_true, y_pred=y_pred)

            out_df = pd.DataFrame({
                "scenario": df["scenario"].astype("string"),
                target_col: y_true.astype(np.int8, copy=False),
                "y_pred": y_pred,
                "y_score": y_score,
            })
            table = pa.Table.from_pandas(out_df, schema=prediction_schema(target_col), preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(str(out_path), table.schema)
            writer.write_table(table)

            if batch_idx % 25 == 0:
                log.info("%s: processed %d prediction chunks", split_name, batch_idx)
    finally:
        if writer is not None:
            writer.close()

    metrics = running.as_dict()
    metrics["predictions_path"] = str(out_path)
    metrics["split_name"] = split_name
    return metrics
