# Hybrid Early Detection

This folder contains the first interpretable hybrid early-detection scaffold for the thesis.

The main design goal is to avoid a black-box fusion model while still combining multiple
evidence types that matter under prefix-based cross-domain IDS:

- aligned tabular feature interactions,
- prefix-aware temporal context,
- and prototype-style domain-robust similarity.

## Why this folder exists

The earlier early-detection experiments already showed:

- in-domain early signal exists,
- cross-domain early transfer is strongly asymmetric,
- transfer learning is conditional and can become negative.

The hybrid workflow asks a more specific follow-up question:

- can a modular, evidence-adaptive model combine different signal types more robustly
  than a single branch, while still remaining interpretable?

## First implementation strategy

The current repository stack is based on `scikit-learn` and `xgboost`, not PyTorch.
So the first hybrid version is intentionally a **non-black-box scaffold** rather than
a full deep end-to-end architecture.

It implements three explicit branches:

1. `tabular`
   A tabular interaction branch over the aligned shared feature space.
2. `temporal`
   A prefix-aware branch using the aligned shared features plus lightweight temporal
   context derived from ordered prefixes.
3. `prototype`
   A prototype similarity branch that compares each example to benign and malicious
   class centroids in a standardized aligned feature space.

These branches are fused by a transparent gating rule based on:

- available evidence (`evidence_progress`),
- per-branch confidence,
- branch agreement,
- and prototype margin.

This means the final prediction can be inspected row-by-row:

- which branch contributed most,
- whether the gate preferred early temporal evidence,
- whether prototype similarity dominated under uncertainty,
- and how branch weights shifted as more evidence became available.

## Relation to the planned architecture

This scaffold is the thesis-safe starting point for the larger hybrid idea:

- the `temporal` branch is the first proxy for a later TCN-style branch,
- the `tabular` branch is the first proxy for a later FT-Transformer-style branch,
- the `prototype` branch already matches the intended prototype-based component,
- the current gate is intentionally interpretable rather than learned end-to-end.

If the scaffold proves useful, the branch internals can later be upgraded while
preserving the same experimental interface and analysis outputs.

## Current scope

The first implemented runner focuses on **cross-domain early detection** because that is
the most natural setting for:

- aligned heterogeneous features,
- explicit domain-shift analysis,
- and interpretable branch weighting under prefix scarcity.

## Files

- `hybrid_early_detection_common.py`
  Shared loaders, branch definitions, temporal feature construction, gating logic,
  and evaluation helpers.
- `run_hybrid_cross_domain_early_detection.py`
  Runs the interpretable hybrid source-only cross-domain early-detection baseline.
- `hybrid_torch_common.py`
  Full Torch implementation with a TCN branch, a closer FT-Transformer tabular branch,
  a prototypical branch, and learned adaptive gating.
- `run_hybrid_full_cross_domain_early_detection.py`
  Runs the full Torch hybrid source-only cross-domain early-detection model.
- `analyze_hybrid_runs.py`
  Aggregates hybrid run outputs and produces branch-weight and performance plots.

## What makes this not a black box

This scaffold is designed to support analysis, not just scores.

Each prediction stores:

- branch probabilities,
- branch weights,
- evidence progress,
- branch agreement,
- and the final fused probability.

That makes the following analyses possible later:

- branch-weight vs prefix-fraction plots,
- branch-weight vs domain-direction plots,
- prototype-margin vs transfer-success plots,
- and ablations where one branch or one gate factor is removed.

## Recommended next step after this scaffold

Once the first hybrid runner is producing outputs cleanly, my next work will focus on the following:

1. compare hybrid vs existing RF/MLP cross-domain baselines,
2. add ablations:
   - no temporal branch,
   - no prototype branch,
   - uniform averaging instead of adaptive gating,
3. analyze which branch dominates at small vs large prefix fractions,
4. decide whether a deeper temporal or transformer branch is worth adding later.

## Full architecture status

The folder now contains two levels of implementation:

1. `run_hybrid_cross_domain_early_detection.py`
   The earlier thesis-safe interpretable scaffold.
2. `run_hybrid_full_cross_domain_early_detection.py`
   The full Torch path with:
   - a real TCN temporal branch,
   - a closer FT-Transformer tabular branch,
   - a prototypical branch,
   - and learned adaptive gating over explicit evidence/confidence inputs.

The full path is the correct runner when the goal is to implement the architecture
as originally proposed rather than only approximate it with a lightweight baseline.


