# Cross-Domain Early Detection

This folder contains the cross-domain early-detection experiments for the thesis.

## Goal

Train on one dataset, evaluate on the other, and ask not only whether transfer works at all, but how much malicious signal is recoverable when only an early prefix of the target-domain evidence is visible.

The setup combines:

- the curated aligned feature subset from `feature_alignment/comparison_outputs/aligned_features_curated.csv`
- the cross-domain train-on-one / test-on-the-other logic from `cross_domain_shift`
- the corrected early-prefix evaluation logic from `early_detection/in_domain_early_detection`

## Target-side meaning of "early"

Cross-domain early detection is defined on the target dataset only.

- When the target is `IoT-23`, evaluation uses scenario-wise temporal prefixes after sorting by `scenario` and `ts`.
- When the target is `UNSW-NB15`, evaluation uses ordered prefixes after sorting by `id`.

This means the two directions remain methodologically analogous, but not temporally identical:

- `UNSW-NB15 -> IoT-23` tests transfer into true scenario-temporal early detection.
- `IoT-23 -> UNSW-NB15` tests transfer into an ordered-prefix early-observation setting.

## Scripts

- `run_cross_domain_early_detection.py`
  Random Forest cross-domain early-detection runner
- `run_cross_domain_mlp_early_detection.py`
  MLP cross-domain early-detection runner
- `analyze_cross_domain_runs.py`
  selects matched RF/MLP runs and generates comparison plots and summary tables
- `run_cross_domain_multi_seed.py`
  launches the matched RF/MLP cross-domain configurations across multiple seeds
- `analyze_cross_domain_multi_seed.py`
  aggregates the repeated-seed cross-domain runs into confidence intervals, paired deltas, and curve-level summaries
- `run_cross_domain_sensitivity.py`
  launches targeted source-size and eval-cap sensitivity experiments against the repeated baseline
- `analyze_cross_domain_sensitivity.py`
  compares the targeted sensitivity runs against the matched repeated baseline using seed-matched summaries and delta plots
- `cross_domain_early_detection_common.py`
  shared aligned-loading, prefix-evaluation, and artifact-writing utilities

## Output structure

Each run directory writes:

- `run_config.json`
- `overall_fraction_summary.csv`
- `overall_iot23_scenario_summary.csv` when `IoT-23` is a target
- `overall_unsw_attack_cat_summary.csv` when `UNSW-NB15` is a target

Each direction writes its own subdirectory:

- `iot23_to_unsw/`
- `unsw_to_iot23/`

Inside each direction:

- model artifact (`rf_pipeline.joblib` or `mlp_pipeline.joblib`)
- `used_features.csv`
- `feature_importance.csv` for RF
- `val/`
- `test/`

Each split folder contains:

- `fraction_summary.csv`
- per-fraction prediction parquets
- target-specific detail tables
- `first_true_positive_fraction.csv`
- for IoT-23 targets also `detection_latency_by_fraction.csv`

## Analyzer outputs

Running `analyze_cross_domain_runs.py` writes:

- `all_run_fraction_summaries.csv`
- `run_manifest.csv`
- `selected_best_run_fraction_summaries.csv`
- `selection_manifest.csv`
- `curve_level_summary.csv`
- `low_fraction_summary.csv`
- direction-specific metric plots under `fixed_analyzer/plots/`
- IoT-23 scenario heatmaps for the selected `UNSW -> IoT-23` runs
- UNSW attack-category heatmaps for the selected `IoT-23 -> UNSW` runs

Running `analyze_cross_domain_multi_seed.py` writes:

- `run_inventory.csv`
- `all_seed_test_rows.csv`
- `per_fraction_summary_stats.csv`
- `curve_level_seed_summary.csv`
- `curve_level_summary_stats.csv`
- `paired_fraction_deltas.csv`
- `paired_curve_deltas.csv`
- `run_matching_validation.csv`
- confidence-band plots and paired-delta plots under `cross_domain_multi_seed_analyzer/plots/`

Running `analyze_cross_domain_sensitivity.py` writes:

- `baseline_run_inventory.csv`
- `sensitivity_run_inventory.csv`
- `comparison_test_rows.csv`
- `per_fraction_summary_stats.csv`
- `curve_level_seed_summary.csv`
- `curve_level_summary_stats.csv`
- `paired_fraction_deltas.csv`
- `paired_curve_deltas.csv`
- `matching_validation.csv`
- study-specific baseline-vs-sensitivity plots and delta plots under `sensitivity_analyzer/plots/`

## Recommended commands

Run from the thesis root.

Safe first smoke test:

```powershell
.\.venv313\Scripts\python.exe .\early_detection\cross_domain_early_detection\run_cross_domain_early_detection.py `
  --iot_data_dir .\Datasets\IoT23\processed_test_sample\iot23 `
  --direction iot23_to_unsw `
  --out_dir .\early_detection\cross_domain_early_detection\smoke_rf `
  --fractions 0.10 1.00 `
  --iot_train_max_rows 1000 `
  --unsw_eval_max_rows 500 `
  --rf_n_estimators 50 `
  --rf_max_depth 8
```

Random Forest:

```powershell
.\.venv313\Scripts\python.exe .\early_detection\cross_domain_early_detection\run_cross_domain_early_detection.py `
  --direction iot23_to_unsw `
  --out_dir .\early_detection\cross_domain_early_detection\outputs_rf_exp1 `
  --fractions 0.02 0.05 0.10 0.20 0.50 1.00 `
  --iot_train_max_rows 100000 `
  --unsw_train_max_rows 100000 `
  --iot_eval_max_rows_per_scenario 50000 `
  --unsw_eval_max_rows 30000
```

MLP:

```powershell
.\.venv313\Scripts\python.exe .\early_detection\cross_domain_early_detection\run_cross_domain_mlp_early_detection.py `
  --direction iot23_to_unsw `
  --out_dir .\early_detection\cross_domain_early_detection\outputs_mlp_exp1 `
  --fractions 0.02 0.05 0.10 0.20 0.50 1.00 `
  --iot_train_max_rows 100000 `
  --unsw_train_max_rows 100000 `
  --iot_eval_max_rows_per_scenario 50000 `
  --unsw_eval_max_rows 30000
```

Analyzer:

```powershell
.\.venv313\Scripts\python.exe .\early_detection\cross_domain_early_detection\analyze_cross_domain_runs.py `
  --runs_dir .\early_detection\cross_domain_early_detection `
  --out_dir .\early_detection\cross_domain_early_detection\fixed_analyzer
```

## Multi-seed repeated baseline

Run the matched 10-seed RF/MLP grid:

```powershell
.\.venv313\Scripts\python.exe .\early_detection\cross_domain_early_detection\run_cross_domain_multi_seed.py `
  --out_dir .\early_detection\cross_domain_early_detection\multiple_seeds_test `
  --seeds 1 2 3 4 5 6 7 8 9 10 `
  --directions iot23_to_unsw unsw_to_iot23 `
  --models rf mlp `
  --fractions 0.02 0.05 0.10 0.20 0.50 1.00 `
  --iot_train_max_rows 100000 `
  --unsw_train_max_rows 100000 `
  --iot_eval_max_rows_per_scenario 50000 `
  --unsw_eval_max_rows 30000
```

If you need to resume an interrupted batch, rerun the same command. Existing completed run folders are skipped unless `--overwrite_existing` is supplied.

Analyze the repeated-seed outputs:

```powershell
.\.venv313\Scripts\python.exe .\early_detection\cross_domain_early_detection\analyze_cross_domain_multi_seed.py `
  --runs_dir .\early_detection\cross_domain_early_detection\multiple_seeds_test `
  --out_dir .\early_detection\cross_domain_early_detection\cross_domain_multi_seed_analyzer
```

## Targeted sensitivity studies

Run the recommended 3-seed source-size and eval-cap sensitivity batch:

```powershell
.\.venv313\Scripts\python.exe .\early_detection\cross_domain_early_detection\run_cross_domain_sensitivity.py `
  --out_dir .\early_detection\cross_domain_early_detection\sensitivity_tests `
  --studies size eval_cap `
  --directions iot23_to_unsw unsw_to_iot23 `
  --models rf mlp `
  --seeds 42 123 456 `
  --fractions 0.02 0.05 0.10 0.20 0.50 1.00 `
  --iot_train_max_rows 100000 `
  --unsw_train_max_rows 100000 `
  --iot_eval_max_rows_per_scenario 50000 `
  --unsw_eval_max_rows 30000 `
  --iot_train_large_rows 200000 `
  --unsw_train_large_rows 175341 `
  --iot_eval_large_rows_per_scenario 75000 `
  --unsw_eval_large_rows 50000
```

Analyze the targeted sensitivity outputs against the repeated baseline:

```powershell
.\.venv313\Scripts\python.exe .\early_detection\cross_domain_early_detection\analyze_cross_domain_sensitivity.py `
  --baseline_dir .\early_detection\cross_domain_early_detection\multiple_seeds_test `
  --sensitivity_dir .\early_detection\cross_domain_early_detection\sensitivity_tests `
  --out_dir .\early_detection\cross_domain_early_detection\sensitivity_analyzer
```
