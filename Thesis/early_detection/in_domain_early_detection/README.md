# In-Domain Early Detection

This folder contains the first early-detection experiment for the thesis.

## Goal

Establish an in-domain IoT-23 early-detection baseline before comparing it to cross-domain early detection.

The experiment uses:

- normal IoT-23 `train.parquet` for fitting
- temporally ordered prefixes within each validation and test scenario for evaluation
- binary intrusion detection (`label_binary`)

## What "early" means here

This setup does **not** claim packet-level real-time detection.

Instead, early detection is operationalized as **partial temporal observation of each scenario**:

- sort each scenario by `ts`
- keep only the first `k%` of rows for `k in {0.1, 0.2, 0.5, 1.0}` by default
- evaluate how performance changes as more of the scenario is observed

This framing is consistent with the current processed IoT-23 flow data and is suitable for the thesis methods section.

## Script

- `run_iot23_in_domain_early_detection.py`

## Outputs

Running the script writes:

- `run_config.json`
- `rf_pipeline.joblib`
- `overall_fraction_summary.csv`
- `overall_scenario_summary.csv`
- `val/fraction_summary.csv`
- `test/fraction_summary.csv`
- per-fraction prediction parquet files
- per-fraction scenario metric CSV files
- detection-latency style summaries:
  - `detection_latency_by_fraction.csv`
  - `first_true_positive_fraction.csv`

## Recommended test-sample command

From the thesis repository root:

```powershell
.\.venv313\Scripts\python.exe .\early_detection\in_domain_early_detection\run_iot23_in_domain_early_detection.py `
  --data_dir .\Datasets\IoT23\processed_test_sample\iot23 `
  --out_dir .\early_detection\in_domain_early_detection\outputs_iot23_sample
```

## Recommended full-data command

```powershell
.\.venv313\Scripts\python.exe .\early_detection\in_domain_early_detection\run_iot23_in_domain_early_detection.py `
  --data_dir .\Datasets\IoT23\processed_full\iot23 `
  --out_dir .\early_detection\in_domain_early_detection\outputs_iot23_full `
  --train_max_rows 300000 `
  --eval_max_rows_per_scenario 50000
```

The row caps are optional but practical for the first full run.
