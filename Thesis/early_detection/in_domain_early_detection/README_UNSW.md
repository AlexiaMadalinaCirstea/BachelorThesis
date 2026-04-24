# In-Domain Early Detection on UNSW-NB15

This is the UNSW-NB15 counterpart to the IoT-23 in-domain early-detection baseline.

## Important methodological note

UNSW-NB15 does not come with IoT-23-style scenario identifiers in the standard training/testing CSVs.

So for UNSW-NB15, early detection is operationalized as:

- sort each split by `id`
- create a stratified validation split from the training CSV by `label`
- sort both the training and validation splits by `id`
- evaluate prefix fractions of the ordered validation and test tables

This is not identical to the scenario-prefix protocol used for IoT-23, but it is a more defensible in-domain ordered-observation baseline for the standard UNSW-NB15 split files than taking the tail of the CSV as validation.

## Script

- `run_unsw_in_domain_early_detection.py`

## Default files

- training: `Datasets/UNSW-NB15/UNSW-NB15 dataset/CSV Files/Training and Testing Sets/UNSW_NB15_training-set.csv`
- testing: `Datasets/UNSW-NB15/UNSW-NB15 dataset/CSV Files/Training and Testing Sets/UNSW_NB15_testing-set.csv`

## Example run

```powershell
.\.venv313\Scripts\python.exe .\early_detection\in_domain_early_detection\run_unsw_in_domain_early_detection.py `
  --out_dir .\early_detection\in_domain_early_detection\outputs_unsw_exp1 `
  --train_max_rows 100000 `
  --eval_max_rows 20000 `
  --rf_n_estimators 150 `
  --rf_max_depth 12 `
  --rf_n_jobs 1
```
