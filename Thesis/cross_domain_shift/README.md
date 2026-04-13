# Cross-Domain Shift

This folder contains the cross-domain shift experiments for the thesis.

## Goal

Train on one dataset and test on the other using the curated aligned feature subset between IoT-23 and UNSW-NB15. And vice versa.

## Default setup

- IoT-23 train: `Datasets/IoT23/processed_full/iot23/train.parquet`
- IoT-23 test: `Datasets/IoT23/processed_full/iot23/test.parquet`
- UNSW-NB15 train: `Datasets/UNSW-NB15/UNSW-NB15 dataset/CSV Files/Training and Testing Sets/UNSW_NB15_training-set.csv`
- UNSW-NB15 test: `Datasets/UNSW-NB15/UNSW-NB15 dataset/CSV Files/Training and Testing Sets/UNSW_NB15_testing-set.csv`
- Curated features: `feature_alignment/comparison_outputs/aligned_features_curated.csv`
