# Transfer Learning

This folder contains the transfer-learning experiments that extend the aligned cross-domain shift work between IoT-23 and UNSW-NB15.


The code in this folder and what is the intended use:

1. `run_transfer_learning.py`
   baseline transfer-learning experiment with three conditions:
   - `source_only`
   - `target_only`
   - `transfer_learning`
2. `analyze_thresholds.py`
   diagnostic threshold sweep for runs where default thresholding may hide collapse or class-bias behavior
3. `transfer_learning_updated_recipe.py`
   - balanced target sampling
   - target fit/calibration split
   - threshold selection on the calibration split
   - reduced source-stage dominance
   - stronger target adaptation
4. `plot_updated_recipe_results.py`
   creates the figures from the updated recipe summary
5. `finalize_transfer_learning_results.py`
   creates tables and a verification report from the updated recipe outputs

## The final result

The final transfer-learning result should be taken from:

- `transfer_learning/outputs_updated_recipe`

The older output folders should be looked at as valuable as diagnostic stages:

- `outputs`
- `outputs_balanced`
- `outputs_balanced_probs`
- `threshold_analysis`

## Directions I used to test

- `IoT-23 -> UNSW-NB15`
- `UNSW-NB15 -> IoT-23`

## Shared feature space

All scripts reuse the curated aligned feature subset defined in:

- `feature_alignment/comparison_outputs/aligned_features_curated.csv`

By default only accepted aligned features are used.

## Per-run artifacts

Each run directory writes:

- `metrics.json`
- `predictions.csv`
- `confusion_matrix.csv`
- `feature_importance.csv`
- `threshold_summary.csv` for the updated recipe

Transfer-learning runs additionally write:

- `feature_importance_pretrain.csv`

Top-level outputs include:

- `run_config.json`
- `transfer_learning_summary.csv` for the baseline script
- `updated_recipe_summary.csv` for the updated recipe

## Outputs

Running the finalization step writes:

- `transfer_learning/final_tables/transfer_learning_main_results_wide.csv`
- `transfer_learning/final_tables/transfer_learning_gain_vs_target_only.csv`
- `transfer_learning/final_tables/transfer_learning_key_findings.csv`
- `transfer_learning/final_tables/transfer_learning_feature_shift_top3.csv`
- `transfer_learning/final_tables/transfer_learning_verification_report.json`

## Recommended commands

From the thesis repository root:

```powershell
.\.venv313\Scripts\python.exe .\transfer_learning\transfer_learning_updated_recipe.py `
  --out_dir .\transfer_learning\outputs_updated_recipe `
  --target_fractions 0.05 0.10 0.25 0.50 1.0 `
  --iot_train_max_rows 150000 `
  --iot_test_max_rows 50000 `
  --unsw_train_max_rows 150000 `
  --unsw_test_max_rows 30000 `
  --balance_target_train `
  --target_balance_ratio 1.0
```

```powershell
.\.venv313\Scripts\python.exe .\transfer_learning\plot_updated_recipe_results.py `
  --summary_csv .\transfer_learning\outputs_updated_recipe\updated_recipe_summary.csv `
  --out_dir .\transfer_learning\figures
```

```powershell
.\.venv313\Scripts\python.exe .\transfer_learning\finalize_transfer_learning_results.py `
  --summary_csv .\transfer_learning\outputs_updated_recipe\updated_recipe_summary.csv `
  --run_config .\transfer_learning\outputs_updated_recipe\run_config.json `
  --outputs_dir .\transfer_learning\outputs_updated_recipe `
  --out_dir .\transfer_learning\final_tables
```

