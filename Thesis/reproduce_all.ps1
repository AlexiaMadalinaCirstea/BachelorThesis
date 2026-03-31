$ErrorActionPreference = "Stop"

function Run-Step {
    param (
        [string]$StepName,
        [string]$Command
    )

    Write-Host ""
    Write-Host "==================================================" -ForegroundColor Cyan
    Write-Host $StepName -ForegroundColor Yellow
    Write-Host "Command: $Command" -ForegroundColor DarkGray
    Write-Host "==================================================" -ForegroundColor Cyan

    Invoke-Expression $Command

    Write-Host ""
    Write-Host "[DONE] $StepName" -ForegroundColor Green
}

Write-Host ""
Write-Host "##################################################" -ForegroundColor Cyan
Write-Host "Bachelor Thesis Full Reproducibility Pipeline" -ForegroundColor Cyan
Write-Host "##################################################" -ForegroundColor Cyan


# LOCAL SAMPLE GENERATION

Run-Step `
    -StepName "Stage 1: Create local IoT-23 sample" `
    -Command 'python local_test.py --data_dir "..\Datasets\IoT23\iot_23_datasets_full\opt\Malware-Project\BigDataset\IoTScenarios" --out_dir "..\Datasets\IoT23\test_sample" --n_rows 2000 --seed 42 --force'


# IOT-23 PREPROCESSING

Run-Step `
    -StepName "Stage 2: Preprocess IoT-23 sample" `
    -Command 'python data_preprocessing/data_prep_iot23.py --data_dir "Datasets\IoT23\test_sample" --out_dir "Datasets\IoT23\processed_test_sample" --sample 1.0 --seed 42'

# RANDOM FOREST BASELINE

Run-Step `
    -StepName "Stage 3: Train Random Forest baseline" `
    -Command 'python data_preprocessing/train_baseline_rf.py --data_dir "Datasets\IoT23\processed_test_sample\iot23" --out_dir "Datasets\IoT23\processed_test_sample\iot23\rf_baseline" --target_col label_binary --seed 42'

Run-Step `
    -StepName "Stage 4: Evaluate Random Forest baseline" `
    -Command 'python data_preprocessing/evaluate.py --pred_file "Datasets\IoT23\processed_test_sample\iot23\rf_baseline\rf_test_predictions.parquet" --out_dir "Datasets\IoT23\processed_test_sample\iot23\rf_baseline\eval" --task binary --y_true_col label_binary --y_pred_col y_pred --y_score_col y_score'

# XGBOOST BASELINE

Run-Step `
    -StepName "Stage 5: Train XGBoost baseline" `
    -Command 'python data_preprocessing/train_baseline_xgboost.py --data_dir "Datasets\IoT23\processed_test_sample\iot23" --out_dir "Datasets\IoT23\processed_test_sample\iot23\xgb_baseline" --target_col label_binary --seed 42'

# STAGE 5 — LOSO EVALUATION

Run-Step `
    -StepName "Stage 6: Run LOSO Random Forest" `
    -Command 'python data_preprocessing/loso_rf.py --data_file "Datasets\IoT23\processed_test_sample\iot23\all_flows.parquet" --out_dir "Datasets\IoT23\processed_test_sample\iot23\rf_loso" --target_col label_binary --seed 42'

Run-Step `
    -StepName "Stage 7: Run LOSO XGBoost" `
    -Command 'python data_preprocessing/loso_xgboost.py --data_file "Datasets\IoT23\processed_test_sample\iot23\all_flows.parquet" --out_dir "Datasets\IoT23\processed_test_sample\iot23\xgb_loso" --target_col label_binary --seed 42'

# MODEL COMPARISON

Run-Step `
    -StepName "Stage 8: Compare baseline models" `
    -Command 'python data_preprocessing/compare_baseline_models.py'

# FEATURE SPACE EXPORT

Run-Step `
    -StepName "Stage 9: Export transformed IoT-23 feature space" `
    -Command 'python data_preprocessing/iot23_export_model_feature_space.py'

# STAGE 8 — FEATURE STABILITY

Run-Step `
    -StepName "Stage 10: Compute feature stability" `
    -Command 'python data_preprocessing/compute_feature_stability.py --rf_long "Datasets\IoT23\processed_test_sample\iot23\feature_transfer_analysis\rf_fold_feature_importances_long.csv" --xgb_long "Datasets\IoT23\processed_test_sample\iot23\feature_transfer_analysis\xgb_fold_feature_importances_long.csv" --out_dir "Datasets\IoT23\processed_test_sample\iot23\feature_stability_analysis"'

Run-Step `
    -StepName "Stage 11: Generate feature stability plots" `
    -Command 'python data_preprocessing/plot_feature_stability_full.py --rf_summary "Datasets\IoT23\processed_test_sample\iot23\feature_stability_analysis\rf_feature_stability_summary.csv" --xgb_summary "Datasets\IoT23\processed_test_sample\iot23\feature_stability_analysis\xgb_feature_stability_summary.csv" --out_dir "Datasets\IoT23\processed_test_sample\iot23\feature_stability_plots"'
