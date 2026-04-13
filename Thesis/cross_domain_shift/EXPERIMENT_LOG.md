# Cross-Domain Shift Experiment Log

This file documents the staged cross-domain shift experiments between IoT-23 and UNSW-NB15 using the curated aligned feature subset.

## Main Question

How well do models trained on one dataset transfer to the other once both datasets are reduced to a carefully aligned shared feature space?

## Shared Setup

- Direction 1: `IoT-23 -> UNSW-NB15`
- Direction 2: `UNSW-NB15 -> IoT-23`
- Core aligned subset: 9 accepted features
- Review ablation: `connection_state` mapped from `conn_state` and `state`
- Models used: Random Forest and XGBoost

## Run Inventory

### `outputs`

- Purpose: first successful RF sanity-check run on the accepted 9-feature aligned subset
- Row caps: IoT-23 train/test `50000/20000`, UNSW train/test `50000/20000`
- Main result: transfer already looked strongly asymmetric, with `UNSW -> IoT-23` nearly collapsed

### `outputs_2`

- Purpose: larger RF baseline on the accepted 9-feature aligned subset
- Row caps: IoT-23 train/test `100000/50000`, UNSW train/test `100000/30000`
- Main result: `IoT-23 -> UNSW` improved markedly, but `UNSW -> IoT-23` remained near-total failure

### `outputs_3`

- Purpose: XGBoost comparison at the same scale as `outputs_2`
- Row caps: IoT-23 train/test `100000/50000`, UNSW train/test `100000/30000`
- Main result: XGBoost improved `IoT-23 -> UNSW` further, but still did not rescue `UNSW -> IoT-23`

### `outputs_4`

- Purpose: ablation test that includes the review-feature mapping `connection_state`
- Row caps: IoT-23 train/test `100000/50000`, UNSW train/test `100000/30000`
- Feature count: 10
- Main result: the extra review feature gained some importance but did not improve overall cross-domain transfer enough to justify making it part of the core aligned subset

### `outputs_5`

- Purpose: larger XGBoost follow-up run on the accepted 9-feature aligned subset
- Row caps: IoT-23 train/test `150000/50000`, UNSW train/test `150000/30000`
- Main result: increasing the XGBoost training size did not solve the transfer problem, which strengthens the argument that the failure is not just a data-volume issue

## Thesis-Level Takeaways

- Cross-domain transfer is strongly asymmetric.
- `IoT-23 -> UNSW-NB15` shows partial transfer, especially with XGBoost.
- `UNSW-NB15 -> IoT-23` remains close to complete failure across runs.
- The curated shared schema makes the comparison fairer, but schema alignment alone does not remove domain shift.
- Feature reliance differs by direction and by model, which supports the thesis claim that feature semantics and feature stability matter more than simply adding more training rows.

## Key Files

- Combined summary CSV: [combined_run_summary.csv](/E:/CloneThesis/BachelorThesis/Thesis/cross_domain_shift/combined_run_summary.csv)
- Combined run notes CSV: [combined_run_notes.csv](/E:/CloneThesis/BachelorThesis/Thesis/cross_domain_shift/combined_run_notes.csv)
- Main figure PNG: [cross_domain_shift_progression.png](/E:/CloneThesis/BachelorThesis/Thesis/cross_domain_shift/figures/cross_domain_shift_progression.png)
- Main figure SVG: [cross_domain_shift_progression.svg](/E:/CloneThesis/BachelorThesis/Thesis/cross_domain_shift/figures/cross_domain_shift_progression.svg)

## Regeneration

Run this from the repository root:

```powershell
.\.venv313\Scripts\python.exe cross_domain_shift\plot_experiment_progression.py
```
