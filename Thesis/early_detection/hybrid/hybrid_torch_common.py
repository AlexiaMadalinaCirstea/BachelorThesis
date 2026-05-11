from __future__ import annotations

import copy
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset


try:
    from ..cross_domain_early_detection.cross_domain_early_detection_common import (
        DEFAULT_FRACTIONS,
        IOT23_META_COLS,
        UNSW_META_COLS,
        build_aligned_frame,
        build_feature_mappings,
        downcast_numeric_columns,
        fraction_to_slug,
        infer_column_types,
        load_alignment_table,
        load_iot23_eval_frame,
        load_iot23_source_train,
        load_unsw_frame,
        maybe_sample_rows,
        normalize_categorical_columns,
        prepare_unsw_eval_frame,
        save_json,
        split_unsw_train_val,
        summarize_iot23_scenarios,
        summarize_unsw_attack_categories,
    )
except ImportError:
    THIS_DIR = Path(__file__).resolve().parent
    EARLY_DETECTION_DIR = THIS_DIR.parent
    if str(EARLY_DETECTION_DIR) not in sys.path:
        sys.path.insert(0, str(EARLY_DETECTION_DIR))
    from cross_domain_early_detection.cross_domain_early_detection_common import (
        DEFAULT_FRACTIONS,
        IOT23_META_COLS,
        UNSW_META_COLS,
        build_aligned_frame,
        build_feature_mappings,
        downcast_numeric_columns,
        fraction_to_slug,
        infer_column_types,
        load_alignment_table,
        load_iot23_eval_frame,
        load_iot23_source_train,
        load_unsw_frame,
        maybe_sample_rows,
        normalize_categorical_columns,
        prepare_unsw_eval_frame,
        save_json,
        split_unsw_train_val,
        summarize_iot23_scenarios,
        summarize_unsw_attack_categories,
    )


GROUP_ID_COL = "__group_id"
ORDER_COL = "__order_col"
EVIDENCE_PROGRESS_COL = "evidence_progress"
PREFIX_ROWS_SEEN_COL = "prefix_rows_seen"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_global_seeds(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def compute_metrics(y_true: pd.Series, y_pred: pd.Series | np.ndarray) -> dict[str, float | int]:
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


def add_prefix_metadata(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "scenario" in out.columns:
        out[GROUP_ID_COL] = out["scenario"].astype("string")
        out[ORDER_COL] = pd.to_numeric(out["ts"], errors="coerce")
    elif "id" in out.columns:
        out[GROUP_ID_COL] = pd.Series(["all"] * len(out), index=out.index, dtype="string")
        out[ORDER_COL] = pd.to_numeric(out["id"], errors="coerce")
    else:
        out[GROUP_ID_COL] = pd.Series(["all"] * len(out), index=out.index, dtype="string")
        out[ORDER_COL] = pd.Series(np.arange(len(out)), index=out.index, dtype="float64")

    out = out.sort_values([GROUP_ID_COL, ORDER_COL], kind="mergesort").reset_index(drop=True)
    counts = out.groupby(GROUP_ID_COL, sort=False)[GROUP_ID_COL].transform("count")
    idx = out.groupby(GROUP_ID_COL, sort=False).cumcount() + 1
    out[PREFIX_ROWS_SEEN_COL] = idx.astype(np.int32)
    out[EVIDENCE_PROGRESS_COL] = (idx / counts).astype(np.float32)
    return downcast_numeric_columns(out)


def assert_valid_hybrid_frame(
    df: pd.DataFrame,
    aligned_feature_cols: list[str],
    categorical_cols: list[str],
    numeric_cols: list[str],
    context: str,
    require_both_labels: bool = True,
) -> None:
    required = set(aligned_feature_cols) | {GROUP_ID_COL, ORDER_COL, EVIDENCE_PROGRESS_COL, PREFIX_ROWS_SEEN_COL, "label"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{context}: missing required columns: {missing}")

    if df.empty:
        raise ValueError(f"{context}: dataframe is empty.")

    if require_both_labels and df["label"].nunique() < 2:
        raise ValueError(f"{context}: requires both benign and attack labels.")

    if df[GROUP_ID_COL].isna().any() or df[ORDER_COL].isna().any():
        raise ValueError(f"{context}: found missing group/order metadata after prefix preparation.")

    evidence = pd.to_numeric(df[EVIDENCE_PROGRESS_COL], errors="coerce")
    if evidence.isna().any():
        raise ValueError(f"{context}: evidence_progress contains NaN.")
    if ((evidence < 0.0) | (evidence > 1.0)).any():
        raise ValueError(f"{context}: evidence_progress outside [0, 1].")

    for col in numeric_cols + [EVIDENCE_PROGRESS_COL, PREFIX_ROWS_SEEN_COL]:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if values.isna().all():
            raise ValueError(f"{context}: numeric column '{col}' is entirely NaN.")

    for col in categorical_cols:
        if col not in df.columns:
            raise ValueError(f"{context}: missing categorical column '{col}'.")


def stratified_train_val_split(df: pd.DataFrame, val_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1.")

    train_parts = []
    val_parts = []
    for label_value, group in df.groupby("label", sort=False):
        n_val = max(1, int(round(len(group) * val_fraction)))
        n_val = min(n_val, len(group) - 1)
        val_subset = group.sample(n=n_val, random_state=seed + int(label_value))
        train_subset = group.drop(val_subset.index)
        train_parts.append(train_subset)
        val_parts.append(val_subset)

    train_df = pd.concat(train_parts, ignore_index=False).sort_values([GROUP_ID_COL, ORDER_COL], kind="mergesort").reset_index(drop=True)
    val_df = pd.concat(val_parts, ignore_index=False).sort_values([GROUP_ID_COL, ORDER_COL], kind="mergesort").reset_index(drop=True)
    return train_df, val_df


class HybridTorchPreprocessor:
    def __init__(
        self,
        categorical_cols: list[str],
        numeric_cols: list[str],
        temporal_numeric_cols: list[str],
    ) -> None:
        self.categorical_cols = categorical_cols
        self.numeric_cols = numeric_cols
        self.temporal_numeric_cols = temporal_numeric_cols
        self.category_maps: dict[str, dict[str, int]] = {}
        self.numeric_medians: dict[str, float] = {}
        self.numeric_means: dict[str, float] = {}
        self.numeric_stds: dict[str, float] = {}
        self.temporal_medians: dict[str, float] = {}
        self.temporal_means: dict[str, float] = {}
        self.temporal_stds: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> None:
        norm_df = normalize_categorical_columns(df.copy(), self.categorical_cols)
        for col in self.categorical_cols:
            values = sorted(norm_df[col].astype("string").fillna("missing").unique().tolist())
            self.category_maps[col] = {val: idx + 1 for idx, val in enumerate(values)}

        for col in self.numeric_cols:
            series = pd.to_numeric(norm_df[col], errors="coerce")
            median = float(series.median()) if not series.dropna().empty else 0.0
            filled = series.fillna(median)
            mean = float(filled.mean())
            std = float(filled.std(ddof=0))
            self.numeric_medians[col] = median
            self.numeric_means[col] = mean
            self.numeric_stds[col] = std if std > 1e-6 else 1.0

        for col in self.temporal_numeric_cols:
            series = pd.to_numeric(norm_df[col], errors="coerce")
            median = float(series.median()) if not series.dropna().empty else 0.0
            filled = series.fillna(median)
            mean = float(filled.mean())
            std = float(filled.std(ddof=0))
            self.temporal_medians[col] = median
            self.temporal_means[col] = mean
            self.temporal_stds[col] = std if std > 1e-6 else 1.0

    def transform_frame(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        norm_df = normalize_categorical_columns(df.copy(), self.categorical_cols)
        cat_arrays = []
        for col in self.categorical_cols:
            mapping = self.category_maps[col]
            cat_series = norm_df[col].astype("string").fillna("missing").map(mapping).fillna(0).astype(np.int64)
            cat_arrays.append(cat_series.to_numpy())
        cat_matrix = np.stack(cat_arrays, axis=1) if cat_arrays else np.zeros((len(df), 0), dtype=np.int64)

        num_arrays = []
        for col in self.numeric_cols:
            series = pd.to_numeric(norm_df[col], errors="coerce").fillna(self.numeric_medians[col])
            scaled = ((series - self.numeric_means[col]) / self.numeric_stds[col]).astype(np.float32)
            num_arrays.append(scaled.to_numpy())
        num_matrix = np.stack(num_arrays, axis=1) if num_arrays else np.zeros((len(df), 0), dtype=np.float32)

        temp_arrays = []
        for col in self.temporal_numeric_cols:
            series = pd.to_numeric(norm_df[col], errors="coerce").fillna(self.temporal_medians[col])
            scaled = ((series - self.temporal_means[col]) / self.temporal_stds[col]).astype(np.float32)
            temp_arrays.append(scaled.to_numpy())
        temp_matrix = np.stack(temp_arrays, axis=1) if temp_arrays else np.zeros((len(df), 0), dtype=np.float32)

        if np.isnan(num_matrix).any() or np.isnan(temp_matrix).any():
            raise ValueError("Preprocessor produced NaNs in numeric or temporal matrices.")

        return {
            "categorical": cat_matrix,
            "numeric": num_matrix,
            "temporal_numeric": temp_matrix,
            "evidence_progress": pd.to_numeric(norm_df[EVIDENCE_PROGRESS_COL], errors="coerce").fillna(0.0).to_numpy(np.float32),
            "labels": pd.to_numeric(norm_df["label"], errors="coerce").fillna(0).to_numpy(np.int64),
        }


class HybridSequenceDataset(Dataset):
    def __init__(
        self,
        categorical: np.ndarray,
        numeric: np.ndarray,
        temporal_numeric: np.ndarray,
        evidence_progress: np.ndarray,
        labels: np.ndarray,
        group_ids: np.ndarray,
        seq_len: int,
    ) -> None:
        self.categorical = categorical
        self.numeric = numeric
        self.temporal_numeric = temporal_numeric
        self.evidence_progress = evidence_progress
        self.labels = labels
        self.seq_len = seq_len
        self.sequences = self._build_sequences(group_ids)
        if self.sequences.shape[1] != self.seq_len:
            raise ValueError("Sequence builder produced the wrong sequence length.")

    def _build_sequences(self, group_ids: np.ndarray) -> np.ndarray:
        seqs = np.zeros((len(self.temporal_numeric), self.seq_len, self.temporal_numeric.shape[1]), dtype=np.float32)
        groups: dict[str, list[int]] = {}
        for idx, gid in enumerate(group_ids):
            groups.setdefault(str(gid), []).append(idx)
        for _, indices in groups.items():
            for pos, idx in enumerate(indices):
                start = max(0, pos + 1 - self.seq_len)
                hist = indices[start : pos + 1]
                seq = self.temporal_numeric[hist]
                seqs[idx, -len(seq) :] = seq
        return seqs

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        return {
            "categorical": torch.as_tensor(self.categorical[idx], dtype=torch.long),
            "numeric": torch.as_tensor(self.numeric[idx], dtype=torch.float32),
            "sequence": torch.as_tensor(self.sequences[idx], dtype=torch.float32),
            "evidence_progress": torch.as_tensor(self.evidence_progress[idx], dtype=torch.float32),
            "label": torch.as_tensor(self.labels[idx], dtype=torch.long),
        }


class TCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = self.conv1(x)
        out = out[..., : x.shape[-1]]
        out = self.act(self.norm1(out))
        out = self.dropout(out)
        out = self.conv2(out)
        out = out[..., : x.shape[-1]]
        out = self.act(self.norm2(out))
        out = self.dropout(out)
        return residual + out


class TemporalTCNBranch(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_blocks: int, dropout: float) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([TCNBlock(hidden_dim, 2**i, dropout) for i in range(num_blocks)])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, sequence: Tensor) -> tuple[Tensor, Tensor]:
        x = self.input_proj(sequence)
        x = x.transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        pooled = x[:, :, -1]
        logits = self.head(pooled)
        return logits, pooled


class NumericalFeatureTokenizer(nn.Module):
    """FT-Transformer-style learnable tokenizer for continuous features."""

    def __init__(self, num_features: int, token_dim: int) -> None:
        super().__init__()
        self.num_features = num_features
        self.token_dim = token_dim
        self.weight = nn.Parameter(torch.empty(num_features, token_dim))
        self.bias = nn.Parameter(torch.empty(num_features, token_dim))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        # x: [batch, num_features] -> [batch, num_features, token_dim]
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class CategoricalFeatureTokenizer(nn.Module):
    """FT-Transformer-style embedding tokenizer with shared offset indexing."""

    def __init__(self, cardinalities: list[int], token_dim: int) -> None:
        super().__init__()
        self.cardinalities = cardinalities
        if cardinalities:
            offsets = torch.tensor([0] + cardinalities[:-1], dtype=torch.long).cumsum(0)
            self.register_buffer("offsets", offsets)
            self.embeddings = nn.Embedding(sum(cardinalities) + len(cardinalities), token_dim)
            nn.init.xavier_uniform_(self.embeddings.weight)
        else:
            self.register_buffer("offsets", torch.zeros(0, dtype=torch.long))
            self.embeddings = None

    def forward(self, x: Tensor) -> Tensor:
        if self.embeddings is None:
            return torch.zeros((x.shape[0], 0, 0), device=x.device)
        shifted = x + self.offsets.unsqueeze(0)
        return self.embeddings(shifted)


class ReGLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        a, b = x.chunk(2, dim=-1)
        return a * torch.relu(b)


class FTTransformerBlock(nn.Module):
    """FT-Transformer block with skipped first attention norm and split dropouts."""

    def __init__(
        self,
        token_dim: int,
        num_heads: int,
        attention_dropout: float,
        ffn_dropout: float,
        residual_dropout: float,
        use_attention_norm: bool,
        ffn_multiplier: float = 4 / 3,
    ) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(token_dim) if use_attention_norm else None
        self.attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.attn_residual_dropout = nn.Dropout(residual_dropout)
        self.ffn_norm = nn.LayerNorm(token_dim)

        hidden_dim = int(token_dim * ffn_multiplier)
        self.ffn = nn.Sequential(
            nn.Linear(token_dim, hidden_dim * 2),
            ReGLU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(hidden_dim, token_dim),
        )
        self.ffn_residual_dropout = nn.Dropout(residual_dropout)

    def forward(self, x: Tensor, cls_only_query: bool = False) -> Tensor:
        attn_input = self.attn_norm(x) if self.attn_norm is not None else x
        query = attn_input[:, :1] if cls_only_query else attn_input
        residual = x[:, :1] if cls_only_query else x
        attn_out, _ = self.attn(query, attn_input, attn_input, need_weights=False)
        x = residual + self.attn_residual_dropout(attn_out)
        x = x + self.ffn_residual_dropout(self.ffn(self.ffn_norm(x)))
        return x


class FTTransformerBranch(nn.Module):
    """A closer FT-Transformer implementation for tabular features."""

    def __init__(
        self,
        num_numeric: int,
        cat_cardinalities: list[int],
        token_dim: int,
        depth: int,
        num_heads: int,
        attention_dropout: float,
        ffn_dropout: float,
        residual_dropout: float,
    ) -> None:
        super().__init__()
        self.num_numeric = num_numeric
        self.cat_cardinalities = cat_cardinalities
        self.token_dim = token_dim
        self.num_tokenizer = NumericalFeatureTokenizer(num_numeric, token_dim) if num_numeric > 0 else None
        # +1 per feature for unknown / zero bucket already present in preprocessed indices.
        adjusted_cardinalities = [cardinality + 1 for cardinality in cat_cardinalities]
        self.cat_tokenizer = CategoricalFeatureTokenizer(adjusted_cardinalities, token_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.blocks = nn.ModuleList([
            FTTransformerBlock(
                token_dim=token_dim,
                num_heads=num_heads,
                attention_dropout=attention_dropout,
                ffn_dropout=ffn_dropout,
                residual_dropout=residual_dropout,
                use_attention_norm=layer_idx > 0,
            )
            for layer_idx in range(depth)
        ])
        self.final_norm = nn.LayerNorm(token_dim)
        self.head = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(token_dim, 2),
        )

    def make_parameter_groups(self) -> list[dict[str, object]]:
        zero_wd_params: list[nn.Parameter] = []
        seen: set[int] = set()

        def add_param(param: nn.Parameter | None) -> None:
            if param is None:
                return
            param_id = id(param)
            if param_id not in seen:
                seen.add(param_id)
                zero_wd_params.append(param)

        add_param(self.cls_token)
        for module in [self.num_tokenizer, self.cat_tokenizer, self.final_norm]:
            if module is not None:
                for param in module.parameters():
                    add_param(param)
        for block in self.blocks:
            if block.attn_norm is not None:
                for param in block.attn_norm.parameters():
                    add_param(param)
            for param in block.ffn_norm.parameters():
                add_param(param)
        for name, param in self.named_parameters():
            if name.endswith("bias"):
                add_param(param)

        main_params = [param for param in self.parameters() if id(param) not in seen]
        groups: list[dict[str, object]] = []
        if main_params:
            groups.append({"params": main_params})
        if zero_wd_params:
            groups.append({"params": zero_wd_params, "weight_decay": 0.0})
        return groups

    def forward(self, numeric: Tensor, categorical: Tensor) -> tuple[Tensor, Tensor]:
        tokens = []
        if self.num_tokenizer is not None:
            num_tokens = self.num_tokenizer(numeric)
            tokens.append(num_tokens)
        if categorical.shape[1] > 0:
            tokens.append(self.cat_tokenizer(categorical))
        x = torch.cat(tokens, dim=1) if tokens else torch.zeros((numeric.shape[0], 0, self.token_dim), device=numeric.device)
        cls = self.cls_token.expand(numeric.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        last_block_idx = len(self.blocks) - 1
        for block_idx, block in enumerate(self.blocks):
            x = block(x, cls_only_query=block_idx == last_block_idx)
        x = self.final_norm(x)
        pooled = x[:, 0]
        logits = self.head(pooled)
        return logits, pooled


class PrototypicalBranch(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, dropout: float) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )
        self.register_buffer("prototypes", torch.zeros(2, embed_dim))
        self.register_buffer("prototype_counts", torch.zeros(2))

    def forward(self, flat_features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        embedding = self.encoder(flat_features)
        distances = torch.cdist(embedding, self.prototypes)
        logits = -distances
        margin = torch.abs(distances[:, 0] - distances[:, 1]).unsqueeze(1)
        return logits, embedding, margin

    @torch.no_grad()
    def recompute_prototypes(self, loader: DataLoader, model: "FullHybridEarlyDetectionModel") -> None:
        sum_embeddings = torch.zeros_like(self.prototypes)
        counts = torch.zeros_like(self.prototype_counts)
        model.eval()
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            flat = model.build_flat_prototype_features(batch["numeric"], batch["categorical"])
            embedding = self.encoder(flat)
            labels = batch["label"]
            for label in (0, 1):
                mask = labels == label
                if mask.any():
                    sum_embeddings[label] += embedding[mask].sum(dim=0)
                    counts[label] += mask.sum()
        for label in (0, 1):
            if counts[label] > 0:
                self.prototypes[label] = sum_embeddings[label] / counts[label]
        self.prototype_counts = counts


class AdaptiveGatingNetwork(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def forward(
        self,
        tab_prob: Tensor,
        temp_prob: Tensor,
        proto_prob: Tensor,
        evidence_progress: Tensor,
        proto_margin: Tensor,
    ) -> Tensor:
        tab_conf = torch.abs(tab_prob - 0.5) * 2.0
        temp_conf = torch.abs(temp_prob - 0.5) * 2.0
        proto_conf = torch.abs(proto_prob - 0.5) * 2.0
        inputs = torch.cat(
            [
                evidence_progress.unsqueeze(1),
                tab_conf.unsqueeze(1),
                temp_conf.unsqueeze(1),
                proto_conf.unsqueeze(1),
                torch.abs(tab_prob - temp_prob).unsqueeze(1),
                torch.abs(tab_prob - proto_prob).unsqueeze(1),
                proto_margin,
            ],
            dim=1,
        )
        return torch.softmax(self.net(inputs), dim=1)


class FullHybridEarlyDetectionModel(nn.Module):
    def __init__(
        self,
        num_numeric: int,
        cat_cardinalities: list[int],
        seq_input_dim: int,
        token_dim: int,
        transformer_depth: int,
        transformer_heads: int,
        tcn_hidden_dim: int,
        tcn_blocks: int,
        prototype_embed_dim: int,
        gate_hidden_dim: int,
        dropout: float,
        attention_dropout: float,
        ffn_dropout: float,
        residual_dropout: float,
        disable_tabular_branch: bool = False,
        disable_temporal_branch: bool = False,
        disable_prototype_branch: bool = False,
        uniform_gating: bool = False,
    ) -> None:
        super().__init__()
        self.disable_tabular_branch = disable_tabular_branch
        self.disable_temporal_branch = disable_temporal_branch
        self.disable_prototype_branch = disable_prototype_branch
        self.uniform_gating = uniform_gating
        if all([disable_tabular_branch, disable_temporal_branch, disable_prototype_branch]):
            raise ValueError("At least one branch must remain enabled.")

        self.tabular_branch = FTTransformerBranch(
            num_numeric=num_numeric,
            cat_cardinalities=cat_cardinalities,
            token_dim=token_dim,
            depth=transformer_depth,
            num_heads=transformer_heads,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            residual_dropout=residual_dropout,
        )
        self.temporal_branch = TemporalTCNBranch(
            input_dim=seq_input_dim,
            hidden_dim=tcn_hidden_dim,
            num_blocks=tcn_blocks,
            dropout=dropout,
        )
        flat_dim = num_numeric + sum(min(16, max(4, math.ceil(math.sqrt(card)))) for card in cat_cardinalities)
        self.cat_embeds = nn.ModuleList(
            [nn.Embedding(cardinality + 1, min(16, max(4, math.ceil(math.sqrt(cardinality))))) for cardinality in cat_cardinalities]
        )
        self.prototype_branch = PrototypicalBranch(flat_dim, prototype_embed_dim, dropout)
        self.gating = AdaptiveGatingNetwork(gate_hidden_dim, dropout)

    def make_parameter_groups(self) -> list[dict[str, object]]:
        tabular_groups = self.tabular_branch.make_parameter_groups()
        tabular_param_ids = {id(param) for group in tabular_groups for param in group["params"]}
        remaining_params = [param for param in self.parameters() if id(param) not in tabular_param_ids]
        groups: list[dict[str, object]] = []
        if remaining_params:
            groups.append({"params": remaining_params})
        groups.extend(group for group in tabular_groups if group["params"])
        return groups

    def enabled_mask(self, batch_size: int, device: torch.device) -> Tensor:
        mask = torch.tensor(
            [
                0.0 if self.disable_tabular_branch else 1.0,
                0.0 if self.disable_temporal_branch else 1.0,
                0.0 if self.disable_prototype_branch else 1.0,
            ],
            dtype=torch.float32,
            device=device,
        )
        return mask.unsqueeze(0).expand(batch_size, -1)

    def build_flat_prototype_features(self, numeric: Tensor, categorical: Tensor) -> Tensor:
        cat_parts = []
        for idx, emb in enumerate(self.cat_embeds):
            cat_parts.append(emb(categorical[:, idx]))
        cat_flat = torch.cat(cat_parts, dim=1) if cat_parts else torch.zeros((numeric.shape[0], 0), device=numeric.device)
        return torch.cat([numeric, cat_flat], dim=1)

    def _uniform_or_masked_weights(
        self,
        raw_weights: Tensor,
        enabled_mask: Tensor,
    ) -> Tensor:
        if self.uniform_gating:
            weights = enabled_mask.clone()
        else:
            weights = raw_weights * enabled_mask
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return weights / denom

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        numeric = batch["numeric"]
        categorical = batch["categorical"]
        sequence = batch["sequence"]
        evidence_progress = batch["evidence_progress"]
        batch_size = numeric.shape[0]
        enabled_mask = self.enabled_mask(batch_size, numeric.device)

        tab_logits, _ = self.tabular_branch(numeric, categorical)
        temp_logits, _ = self.temporal_branch(sequence)
        proto_logits, _, proto_margin = self.prototype_branch(self.build_flat_prototype_features(numeric, categorical))

        if self.disable_tabular_branch:
            tab_logits = torch.zeros_like(tab_logits)
        if self.disable_temporal_branch:
            temp_logits = torch.zeros_like(temp_logits)
        if self.disable_prototype_branch:
            proto_logits = torch.zeros_like(proto_logits)
            proto_margin = torch.zeros_like(proto_margin)

        tab_prob = torch.softmax(tab_logits, dim=1)[:, 1]
        temp_prob = torch.softmax(temp_logits, dim=1)[:, 1]
        proto_prob = torch.softmax(proto_logits, dim=1)[:, 1]

        gate_tab_prob = tab_prob if not self.disable_tabular_branch else torch.full_like(tab_prob, 0.5)
        gate_temp_prob = temp_prob if not self.disable_temporal_branch else torch.full_like(temp_prob, 0.5)
        gate_proto_prob = proto_prob if not self.disable_prototype_branch else torch.full_like(proto_prob, 0.5)
        gate_proto_margin = proto_margin if not self.disable_prototype_branch else torch.zeros_like(proto_margin)

        raw_weights = self.gating(gate_tab_prob, gate_temp_prob, gate_proto_prob, evidence_progress, gate_proto_margin)
        weights = self._uniform_or_masked_weights(raw_weights, enabled_mask)

        fused_logits = (
            weights[:, 0:1] * tab_logits
            + weights[:, 1:2] * temp_logits
            + weights[:, 2:3] * proto_logits
        )

        tab_conf = torch.abs(tab_prob - 0.5) * 2.0
        temp_conf = torch.abs(temp_prob - 0.5) * 2.0
        proto_conf = torch.abs(proto_prob - 0.5) * 2.0
        branch_probs = torch.stack([tab_prob, temp_prob, proto_prob], dim=1)
        branch_votes = (branch_probs >= 0.5).float() * enabled_mask
        enabled_counts = enabled_mask.sum(dim=1).clamp_min(1.0)
        attack_votes = branch_votes.sum(dim=1)
        majority_votes = torch.maximum(attack_votes, enabled_counts - attack_votes)
        branch_agreement = torch.where(
            enabled_counts <= 1.0,
            torch.ones_like(enabled_counts),
            majority_votes / enabled_counts,
        )

        return {
            "tab_logits": tab_logits,
            "temp_logits": temp_logits,
            "proto_logits": proto_logits,
            "fused_logits": fused_logits,
            "weights": weights,
            "tab_prob": tab_prob,
            "temp_prob": temp_prob,
            "proto_prob": proto_prob,
            "tab_conf": tab_conf,
            "temp_conf": temp_conf,
            "proto_conf": proto_conf,
            "branch_agreement": branch_agreement,
            "fused_prob": torch.softmax(fused_logits, dim=1)[:, 1],
            "proto_margin": proto_margin.squeeze(1),
            "enabled_mask": enabled_mask,
        }


@dataclass
class HybridTrainingArtifacts:
    model: FullHybridEarlyDetectionModel
    preprocessor: HybridTorchPreprocessor
    categorical_cols: list[str]
    numeric_cols: list[str]
    temporal_numeric_cols: list[str]
    seq_len: int
    history: list[dict[str, float | int | str]]
    best_epoch: int
    best_val_metrics: dict[str, float | int]
    ablation_config: dict[str, bool]


def build_dataset(df: pd.DataFrame, preprocessor: HybridTorchPreprocessor, seq_len: int) -> HybridSequenceDataset:
    transformed = preprocessor.transform_frame(df)
    dataset = HybridSequenceDataset(
        categorical=transformed["categorical"],
        numeric=transformed["numeric"],
        temporal_numeric=transformed["temporal_numeric"],
        evidence_progress=transformed["evidence_progress"],
        labels=transformed["labels"],
        group_ids=df[GROUP_ID_COL].astype("string").to_numpy(),
        seq_len=seq_len,
    )
    return dataset


@torch.no_grad()
def evaluate_model_on_loader(model: FullHybridEarlyDetectionModel, loader: DataLoader) -> dict[str, float]:
    model.eval()
    losses = []
    logits_all = []
    labels_all = []
    ce = nn.CrossEntropyLoss()
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        out = model(batch)
        label = batch["label"]
        loss = ce(out["fused_logits"], label)
        loss = loss + 0.25 * ce(out["tab_logits"], label)
        loss = loss + 0.25 * ce(out["temp_logits"], label)
        loss = loss + 0.25 * ce(out["proto_logits"], label)
        losses.append(float(loss.item()))
        logits_all.append(out["fused_logits"].cpu())
        labels_all.append(label.cpu())

    logits = torch.cat(logits_all, dim=0)
    labels = torch.cat(labels_all, dim=0)
    probs = torch.softmax(logits, dim=1)[:, 1].numpy()
    preds = (probs >= 0.5).astype(np.int64)
    metrics = compute_metrics(pd.Series(labels.numpy()), preds)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics


def train_full_hybrid(
    source_df: pd.DataFrame,
    categorical_cols: list[str],
    numeric_cols: list[str],
    temporal_numeric_cols: list[str],
    seq_len: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    token_dim: int,
    transformer_depth: int,
    transformer_heads: int,
    tcn_hidden_dim: int,
    tcn_blocks: int,
    prototype_embed_dim: int,
    gate_hidden_dim: int,
    dropout: float,
    attention_dropout: float,
    ffn_dropout: float,
    residual_dropout: float,
    seed: int,
    train_val_fraction: float,
    deterministic: bool,
    disable_tabular_branch: bool = False,
    disable_temporal_branch: bool = False,
    disable_prototype_branch: bool = False,
    uniform_gating: bool = False,
) -> HybridTrainingArtifacts:
    set_global_seeds(seed, deterministic)

    assert_valid_hybrid_frame(source_df, categorical_cols + numeric_cols, categorical_cols, numeric_cols, "source_train_full")
    train_df, val_df = stratified_train_val_split(source_df, train_val_fraction, seed)
    assert_valid_hybrid_frame(train_df, categorical_cols + numeric_cols, categorical_cols, numeric_cols, "source_train_split")
    assert_valid_hybrid_frame(val_df, categorical_cols + numeric_cols, categorical_cols, numeric_cols, "source_val_split")

    preprocessor = HybridTorchPreprocessor(categorical_cols, numeric_cols, temporal_numeric_cols)
    preprocessor.fit(train_df)
    train_dataset = build_dataset(train_df, preprocessor, seq_len)
    val_dataset = build_dataset(val_df, preprocessor, seq_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    cat_cardinalities = [len(preprocessor.category_maps[col]) for col in categorical_cols]
    model = FullHybridEarlyDetectionModel(
        num_numeric=len(numeric_cols),
        cat_cardinalities=cat_cardinalities,
        seq_input_dim=len(temporal_numeric_cols),
        token_dim=token_dim,
        transformer_depth=transformer_depth,
        transformer_heads=transformer_heads,
        tcn_hidden_dim=tcn_hidden_dim,
        tcn_blocks=tcn_blocks,
        prototype_embed_dim=prototype_embed_dim,
        gate_hidden_dim=gate_hidden_dim,
        dropout=dropout,
        attention_dropout=attention_dropout,
        ffn_dropout=ffn_dropout,
        residual_dropout=residual_dropout,
        disable_tabular_branch=disable_tabular_branch,
        disable_temporal_branch=disable_temporal_branch,
        disable_prototype_branch=disable_prototype_branch,
        uniform_gating=uniform_gating,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.make_parameter_groups(), lr=learning_rate, weight_decay=weight_decay)
    ce = nn.CrossEntropyLoss()
    history: list[dict[str, float | int | str]] = []
    best_epoch = 0
    best_val_metrics: dict[str, float | int] = {}
    best_state = copy.deepcopy(model.state_dict())
    best_score = float("-inf")

    for epoch in range(1, max(1, epochs) + 1):
        model.train()
        epoch_losses = []
        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            out = model(batch)
            label = batch["label"]
            loss = ce(out["fused_logits"], label)
            loss = loss + 0.25 * ce(out["tab_logits"], label)
            loss = loss + 0.25 * ce(out["temp_logits"], label)
            loss = loss + 0.25 * ce(out["proto_logits"], label)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        model.prototype_branch.recompute_prototypes(eval_loader, model)
        train_metrics = evaluate_model_on_loader(model, eval_loader)
        val_metrics = evaluate_model_on_loader(model, val_loader)
        epoch_record: dict[str, float | int | str] = {
            "epoch": epoch,
            "train_loss_epoch_mean": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
            "train_loss_eval": float(train_metrics["loss"]),
            "train_f1_attack": float(train_metrics["f1_attack"]),
            "train_recall_attack": float(train_metrics["recall_attack"]),
            "val_loss": float(val_metrics["loss"]),
            "val_f1_attack": float(val_metrics["f1_attack"]),
            "val_recall_attack": float(val_metrics["recall_attack"]),
            "val_f1_macro": float(val_metrics["f1_macro"]),
            "val_accuracy": float(val_metrics["accuracy"]),
        }
        history.append(epoch_record)

        score = float(val_metrics["f1_attack"])
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_val_metrics = dict(val_metrics)
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.prototype_branch.recompute_prototypes(eval_loader, model)

    return HybridTrainingArtifacts(
        model=model,
        preprocessor=preprocessor,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
        temporal_numeric_cols=temporal_numeric_cols,
        seq_len=seq_len,
        history=history,
        best_epoch=best_epoch,
        best_val_metrics=best_val_metrics,
        ablation_config={
            "disable_tabular_branch": disable_tabular_branch,
            "disable_temporal_branch": disable_temporal_branch,
            "disable_prototype_branch": disable_prototype_branch,
            "uniform_gating": uniform_gating,
        },
    )


@torch.no_grad()
def predict_with_full_hybrid(df: pd.DataFrame, artifacts: HybridTrainingArtifacts, batch_size: int) -> pd.DataFrame:
    dataset = build_dataset(df, artifacts.preprocessor, artifacts.seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    artifacts.model.eval()

    rows = []
    offset = 0
    for batch in loader:
        batch_size_now = batch["label"].shape[0]
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        out = artifacts.model(batch)
        fused_prob = out["fused_prob"].cpu().numpy()
        y_pred = (fused_prob >= 0.5).astype(np.int64)
        branch_winner = np.argmax(out["weights"].cpu().numpy(), axis=1)
        rows.append(
            pd.DataFrame(
                {
                    "tabular_prob": out["tab_prob"].cpu().numpy(),
                    "temporal_prob": out["temp_prob"].cpu().numpy(),
                    "prototype_prob": out["proto_prob"].cpu().numpy(),
                    "tabular_confidence": out["tab_conf"].cpu().numpy(),
                    "temporal_confidence": out["temp_conf"].cpu().numpy(),
                    "prototype_confidence": out["proto_conf"].cpu().numpy(),
                    "tabular_weight": out["weights"][:, 0].cpu().numpy(),
                    "temporal_weight": out["weights"][:, 1].cpu().numpy(),
                    "prototype_weight": out["weights"][:, 2].cpu().numpy(),
                    "prototype_margin": out["proto_margin"].cpu().numpy(),
                    "branch_agreement": out["branch_agreement"].cpu().numpy(),
                    "branch_winner": branch_winner,
                    "fused_prob": fused_prob,
                    "y_pred": y_pred,
                },
                index=np.arange(offset, offset + batch_size_now),
            )
        )
        offset += batch_size_now
    return pd.concat(rows).sort_index().reset_index(drop=True)


def summarize_prediction_diagnostics(pred_df: pd.DataFrame) -> dict[str, float]:
    branch_winner = pd.to_numeric(pred_df["branch_winner"], errors="coerce").fillna(-1).astype(int)
    return {
        "mean_tabular_weight": float(pred_df["tabular_weight"].mean()),
        "mean_temporal_weight": float(pred_df["temporal_weight"].mean()),
        "mean_prototype_weight": float(pred_df["prototype_weight"].mean()),
        "std_tabular_weight": float(pred_df["tabular_weight"].std(ddof=0)),
        "std_temporal_weight": float(pred_df["temporal_weight"].std(ddof=0)),
        "std_prototype_weight": float(pred_df["prototype_weight"].std(ddof=0)),
        "mean_tabular_confidence": float(pred_df["tabular_confidence"].mean()),
        "mean_temporal_confidence": float(pred_df["temporal_confidence"].mean()),
        "mean_prototype_confidence": float(pred_df["prototype_confidence"].mean()),
        "mean_branch_agreement": float(pred_df["branch_agreement"].mean()),
        "mean_prototype_margin": float(pred_df["prototype_margin"].mean()),
        "tabular_branch_win_rate": float((branch_winner == 0).mean()),
        "temporal_branch_win_rate": float((branch_winner == 1).mean()),
        "prototype_branch_win_rate": float((branch_winner == 2).mean()),
    }


def evaluate_full_hybrid_iot23_target_split(
    split_name: str,
    df: pd.DataFrame,
    artifacts: HybridTrainingArtifacts,
    fractions: list[float],
    out_dir: Path,
    eval_batch_size: int,
) -> dict[str, pd.DataFrame]:
    split_dir = out_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    detail_rows = []
    diagnostics_rows = []

    first_tp_rows = []
    for scenario, group in df.groupby("scenario", sort=False):
        first_fraction = None
        for fraction in fractions:
            keep = max(1, int(len(group) * fraction))
            prefix_df = group.iloc[:keep].copy()
            pred_only = predict_with_full_hybrid(prefix_df, artifacts, eval_batch_size)
            joined = pd.concat([prefix_df.reset_index(drop=True), pred_only], axis=1)
            if ((joined["label"] == 1) & (joined["y_pred"] == 1)).any():
                first_fraction = float(fraction)
                break
        first_tp_rows.append({"scenario": scenario, "first_true_positive_fraction": first_fraction})
    first_tp_df = pd.DataFrame(first_tp_rows)
    first_tp_df.to_csv(split_dir / "first_true_positive_fraction.csv", index=False)

    for fraction in fractions:
        parts = []
        for _, group in df.groupby("scenario", sort=False):
            keep = max(1, int(len(group) * fraction))
            parts.append(group.iloc[:keep])
        prefix_df = pd.concat(parts, ignore_index=True).copy()
        pred_only = predict_with_full_hybrid(prefix_df, artifacts, eval_batch_size)
        pred_df = pd.concat([prefix_df.reset_index(drop=True), pred_only], axis=1)
        pred_df["y_score"] = pred_df["fused_prob"]

        metrics = compute_metrics(pred_df["label"], pred_df["y_pred"])
        diagnostics = summarize_prediction_diagnostics(pred_df)
        scenario_df = summarize_iot23_scenarios(pred_df).merge(first_tp_df, on="scenario", how="left")
        scenario_df["split"] = split_name
        scenario_df["fraction"] = fraction
        slug = fraction_to_slug(fraction)
        pred_df.to_parquet(split_dir / f"predictions_frac_{slug}.parquet", index=False)
        scenario_df.to_csv(split_dir / f"scenario_metrics_frac_{slug}.csv", index=False)
        summary_rows.append(
            {
                "split": split_name,
                "fraction": fraction,
                "rows_evaluated": int(len(pred_df)),
                "n_scenarios": int(pred_df["scenario"].nunique()),
                **diagnostics,
                **metrics,
            }
        )
        diagnostics_rows.append({"split": split_name, "fraction": fraction, **diagnostics})
        detail_rows.append(scenario_df)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(split_dir / "fraction_summary.csv", index=False)
    pd.DataFrame(diagnostics_rows).to_csv(split_dir / "branch_diagnostics.csv", index=False)
    details_df = pd.concat(detail_rows, ignore_index=True)
    details_df.to_csv(split_dir / "scenario_metrics_all_fractions.csv", index=False)
    return {"summary": summary_df, "details": details_df}


def evaluate_full_hybrid_unsw_target_split(
    split_name: str,
    df: pd.DataFrame,
    artifacts: HybridTrainingArtifacts,
    fractions: list[float],
    out_dir: Path,
    eval_batch_size: int,
) -> dict[str, pd.DataFrame]:
    split_dir = out_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    full_pred = pd.concat([df.reset_index(drop=True), predict_with_full_hybrid(df, artifacts, eval_batch_size)], axis=1)
    first_tp = None
    for fraction in fractions:
        keep = max(1, int(len(full_pred) * fraction))
        prefix = full_pred.iloc[:keep]
        if ((prefix["label"] == 1) & (prefix["y_pred"] == 1)).any():
            first_tp = float(fraction)
            break
    pd.DataFrame([{"split": split_name, "first_true_positive_fraction": first_tp}]).to_csv(
        split_dir / "first_true_positive_fraction.csv", index=False
    )

    summary_rows = []
    detail_rows = []
    diagnostics_rows = []
    for fraction in fractions:
        keep = max(1, int(len(full_pred) * fraction))
        pred_df = full_pred.iloc[:keep].copy().reset_index(drop=True)
        pred_df["y_score"] = pred_df["fused_prob"]
        metrics = compute_metrics(pred_df["label"], pred_df["y_pred"])
        diagnostics = summarize_prediction_diagnostics(pred_df)
        attack_cat_df = summarize_unsw_attack_categories(pred_df)
        attack_cat_df["split"] = split_name
        attack_cat_df["fraction"] = fraction
        attack_cat_df["first_true_positive_fraction"] = first_tp
        slug = fraction_to_slug(fraction)
        pred_df.to_parquet(split_dir / f"predictions_frac_{slug}.parquet", index=False)
        attack_cat_df.to_csv(split_dir / f"attack_cat_metrics_frac_{slug}.csv", index=False)
        summary_rows.append(
            {
                "split": split_name,
                "fraction": fraction,
                "rows_evaluated": int(len(pred_df)),
                "n_attack_categories": int(pred_df["attack_cat"].nunique()),
                "first_true_positive_fraction": first_tp,
                **diagnostics,
                **metrics,
            }
        )
        diagnostics_rows.append({"split": split_name, "fraction": fraction, **diagnostics})
        detail_rows.append(attack_cat_df)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(split_dir / "fraction_summary.csv", index=False)
    pd.DataFrame(diagnostics_rows).to_csv(split_dir / "branch_diagnostics.csv", index=False)
    details_df = pd.concat(detail_rows, ignore_index=True)
    details_df.to_csv(split_dir / "attack_cat_metrics_all_fractions.csv", index=False)
    return {"summary": summary_df, "details": details_df}
