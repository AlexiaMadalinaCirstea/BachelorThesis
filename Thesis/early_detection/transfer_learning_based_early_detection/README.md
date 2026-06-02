# Transfer-Learning Early Detection

This folder contains the dedicated transfer-learning early-detection study.

This chapter is intentionally different from:

- `in_domain_early_detection`, which asks whether early signal exists within each dataset;
- `cross_domain_early_detection`, which asks whether early signal transfers with no target adaptation.

The present workflow asks a different question:

- when does source pretraining help, fail to help, or harm early target-domain detection once limited target supervision is introduced?

## Core design

Each matched run contains three conditions:

- `source_only`
- `target_only`
- `transfer_adapted`

The main comparison is:

- `transfer_adapted - target_only`

This is the transfer-gain quantity that matters scientifically. It tests whether source pretraining adds value beyond training directly on the same scarce target subset.

## Why the preprocessing is shared

Within each matched run, the MLP feature preprocessor is fit on the union of:

- source train rows used for pretraining
- target train rows used for adaptation / target-only training

This keeps the transformed input space matched across:

- source-only
- target-only
- transfer-adapted

and makes weight transfer into the fine-tuning stage technically clean.

The preprocessor never sees target validation or target test rows.

## Scripts

- `transfer_learning_early_detection_common.py`
  shared loading, preprocessing, MLP training, and evaluation helpers
- `run_transfer_learning_early_detection.py`
  executes one matched transfer-learning early-detection run for one direction, one target-train budget, and one seed
- `run_transfer_learning_multi_seed.py`
  launches the repeated multi-seed matrix across directions and target-train budgets
- `analyze_transfer_learning_multi_seed.py`
  aggregates repeated runs, computes transfer gains, classifies positive/neutral/negative transfer, and writes plots

## Main outputs per run

Each run directory writes:

- `run_config.json`
- `condition_models.joblib`
- `overall_fraction_summary.csv`
- `overall_iot23_scenario_summary.csv` when IoT-23 is the target
- `overall_unsw_attack_cat_summary.csv` when UNSW-NB15 is the target

Each condition gets its own folder:

- `source_only/`
- `target_only/`
- `transfer_adapted/`

Inside each condition:

- `val/fraction_summary.csv`
- `test/fraction_summary.csv`
- per-fraction prediction parquet files
- per-fraction scenario/category metric CSV files

## Recommended single-run command

Run from the thesis root:

```powershell
.\.venv313\Scripts\python.exe .\early_detection\transfer_learning_based_early_detection\run_transfer_learning_early_detection.py `
  --direction iot23_to_unsw `
  --out_dir .\early_detection\transfer_learning_based_early_detection\outputs_iot23_to_unsw_budget5k_seed42 `
  --target_train_rows 5000 `
  --seed 42 `
  --fractions 0.02 0.05 0.10 0.20 0.50 1.00 `
  --source_train_rows 100000 `
  --source_epochs 20 `
  --target_only_epochs 20 `
  --finetune_epochs 10
```

## Recommended multi-seed command

```powershell
.\.venv313\Scripts\python.exe .\early_detection\transfer_learning_based_early_detection\run_transfer_learning_multi_seed.py `
  --out_dir .\early_detection\transfer_learning_based_early_detection\multiple_seeds_test `
  --directions iot23_to_unsw unsw_to_iot23 `
  --target_train_budgets 1000 5000 20000 50000 `
  --seeds 42 123 456 `
  --fractions 0.02 0.05 0.10 0.20 0.50 1.00 `
  --source_train_rows 100000 `
  --source_epochs 20 `
  --target_only_epochs 20 `
  --finetune_epochs 10
```

## Analyzer command

```powershell
.\.venv313\Scripts\python.exe .\early_detection\transfer_learning_based_early_detection\analyze_transfer_learning_multi_seed.py `
  --runs_dir .\early_detection\transfer_learning_based_early_detection\multiple_seeds_test `
  --out_dir .\early_detection\transfer_learning_based_early_detection\multi_seed_analyzer `
  --gain_epsilon 0.005
```

## Main analyzer outputs

The analyzer writes:

- `run_inventory.csv`
- `all_seed_test_rows.csv`
- `per_condition_fraction_summary_stats.csv`
- `per_fraction_transfer_gain_stats.csv`
- `curve_level_seed_summary.csv`
- `curve_level_transfer_gain_stats.csv`
- `curve_level_transfer_gain_classification.csv`
- `fraction_level_transfer_gain_classification.csv`
- condition and gain plots under `plots/`

## Intended chapter claims

This setup is designed to answer:

- whether transfer helps under scarce target supervision;
- whether transfer benefit is direction-dependent;
- whether benefit is concentrated at tiny target prefixes;
- whether transfer becomes neutral or negative as target-train budget increases.
