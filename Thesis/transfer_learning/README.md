# Transfer Learning

This folder contains the transfer-learning experiments that extend the existing cross-domain shift work between IoT-23 and UNSW-NB15.

## Current experiment design

The main script compares three conditions for each transfer direction:

- `source_only`
- `target_only`
- `transfer_learning`

Transfer learning is implemented as staged XGBoost training:

1. pretrain on the source-domain training split
2. continue training on a target-domain training subset
3. evaluate on the target-domain test split

## Directions

- `IoT-23 -> UNSW-NB15`
- `UNSW-NB15 -> IoT-23`

## Shared feature space

The script reuses the curated aligned feature subset built in `feature_alignment/comparison_outputs/aligned_features_curated.csv`.

By default it uses the accepted aligned features only.

## Outputs

Each run writes:

- `metrics.json`
- `predictions.csv`
- `confusion_matrix.csv`
- `feature_importance.csv`

Transfer-learning runs additionally save:

- `feature_importance_pretrain.csv`

The combined summary is written to:

- `transfer_learning/outputs/transfer_learning_summary.csv`

## Example run

From the thesis repository root:

```powershell
.\.venv313\Scripts\python.exe .\transfer_learning\run_transfer_learning.py `
  --out_dir .\transfer_learning\outputs `
  --target_fractions 0.05 0.10 0.25 0.50 1.0 `
  --iot_train_max_rows 150000 `
  --iot_test_max_rows 50000 `
  --unsw_train_max_rows 150000 `
  --unsw_test_max_rows 30000
```


