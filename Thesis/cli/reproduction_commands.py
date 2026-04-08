from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

COMMANDS = {
    "iot_sample_local_test": {
        "label": "IoT-23 · Build test sample",
        "description": "Create a small stratified IoT-23 sample for fast local testing.",
        "cmd": [
            PYTHON,
            "local_test.py",
            "--data_dir", str(ROOT / "Datasets" / "IoT23" / "iot_23_datasets_full" / "opt" / "Malware-Project" / "BigDataset" / "IoTScenarios"),
            "--out_dir", str(ROOT / "Datasets" / "IoT23" / "test_sample"),
            "--n_rows", "2000",
            "--seed", "42",
            "--force",
        ],
    },
    "iot_sample_prep": {
        "label": "IoT-23 · Preprocess test sample",
        "description": "Run preprocessing on the IoT-23 local test sample.",
        "cmd": [
            PYTHON,
            "data_preprocessing/data_prep_iot23.py",
            "--data_dir", str(ROOT / "Datasets" / "IoT23" / "test_sample"),
            "--out_dir", str(ROOT / "Datasets" / "IoT23" / "processed_test_sample"),
            "--sample", "1.0",
            "--seed", "42",
        ],
    },
    "iot_rf_baseline": {
        "label": "IoT-23 · RF baseline",
        "description": "Train Random Forest baseline on processed IoT-23 sample.",
        "cmd": [
            PYTHON,
            "data_preprocessing/train_baseline_rf.py",
            "--data_dir", str(ROOT / "Datasets" / "IoT23" / "processed_test_sample" / "iot23"),
            "--out_dir", str(ROOT / "Datasets" / "IoT23" / "processed_test_sample" / "iot23" / "rf_baseline"),
            "--target_col", "label_binary",
            "--seed", "42",
        ],
    },
    "iot_rf_eval": {
        "label": "IoT-23 · Evaluate RF baseline",
        "description": "Evaluate Random Forest predictions on IoT-23 sample.",
        "cmd": [
            PYTHON,
            "data_preprocessing/evaluate.py",
            "--pred_file", str(ROOT / "Datasets" / "IoT23" / "processed_test_sample" / "iot23" / "rf_baseline" / "rf_test_predictions.parquet"),
            "--out_dir", str(ROOT / "Datasets" / "IoT23" / "processed_test_sample" / "iot23" / "rf_baseline" / "eval"),
            "--task", "binary",
            "--y_true_col", "label_binary",
            "--y_pred_col", "y_pred",
            "--y_score_col", "y_score",
        ],
    },
    "iot_loso_rf": {
        "label": "IoT-23 · LOSO RF",
        "description": "Run Leave-One-Scenario-Out evaluation with Random Forest on IoT-23.",
        "cmd": [
            PYTHON,
            "data_preprocessing/loso_rf.py",
            "--data_file", str(ROOT / "Datasets" / "IoT23" / "processed_test_sample" / "iot23" / "all_flows.parquet"),
            "--out_dir", str(ROOT / "Datasets" / "IoT23" / "processed_test_sample" / "iot23" / "rf_loso"),
            "--target_col", "label_binary",
            "--seed", "42",
        ],
    },
    "iot_full_prep": {
        "label": "IoT-23 · Full preprocessing",
        "description": "Run preprocessing on the full IoT-23 dataset.",
        "cmd": [
            PYTHON,
            "data_preprocessing/data_prep_iot23.py",
            "--data_dir", str(ROOT / "Datasets" / "IoT23" / "iot_23_datasets_full" / "opt" / "Malware-Project" / "BigDataset" / "IoTScenarios"),
            "--out_dir", str(ROOT / "Datasets" / "IoT23" / "processed_full"),
            "--sample", "1.0",
            "--seed", "42",
        ],
    },
    "cross_dataset_eval": {
        "label": "Cross-dataset evaluation",
        "description": "Compare IoT-23 and UNSW-NB15 processed datasets using RF and XGB.",
        "cmd": [
            PYTHON,
            "data_preprocessing/cross_dataset_eval.py",
            "--iot_csv", str(ROOT / "Datasets" / "IoT23" / "processed_full" / "iot23" / "iot23_final.csv"),
            "--unsw_csv", str(ROOT / "Datasets" / "UNSW_NB15" / "processed_full" / "unsw_nb15" / "unsw_final.csv"),
            "--out_dir", str(ROOT / "Datasets" / "cross_dataset_eval"),
            "--models", "rf", "xgb",
            "--drop_cols", "timestamp", "scenario_id",
        ],
    },
    "cross_dataset_plots": {
        "label": "Cross-dataset plots",
        "description": "Generate plots for cross-dataset evaluation.",
        "cmd": [
            PYTHON,
            "data_preprocessing/plot_cross_dataset_results.py",
            "--summary_csv", str(ROOT / "Datasets" / "cross_dataset_eval" / "cross_dataset_summary.csv"),
            "--out_dir", str(ROOT / "Datasets" / "cross_dataset_eval" / "plots"),
        ],
    },
    "iot_feature_stability_plots": {
        "label": "IoT-23 · Feature stability plots",
        "description": "Plot IoT-23 feature stability outputs.",
        "cmd": [
            PYTHON,
            "data_preprocessing/plot_feature_stability_full.py",
            "--rf_summary", str(ROOT / "Datasets" / "IoT23" / "processed_test_sample" / "iot23" / "feature_stability_analysis" / "rf_feature_stability_summary.csv"),
            "--xgb_summary", str(ROOT / "Datasets" / "IoT23" / "processed_test_sample" / "iot23" / "feature_stability_analysis" / "xgb_feature_stability_summary.csv"),
            "--out_dir", str(ROOT / "Datasets" / "IoT23" / "processed_test_sample" / "iot23" / "feature_stability_plots"),
        ],
    },
    "unsw_inspect": {
        "label": "UNSW-NB15 · Inspect dataset",
        "description": "Inspect official UNSW-NB15 train/test files and save summary outputs.",
        "cmd": [
            PYTHON,
            "full_dataset_preprocessing/baseline_improved_UNSWNB15/data_prep_unsw.py",
            "--train_path", str(ROOT / "Datasets" / "UNSW-NB15" / "UNSW-NB15 dataset" / "CSV Files" / "Training and Testing Sets" / "UNSW_NB15_training-set.csv"),
            "--test_path", str(ROOT / "Datasets" / "UNSW-NB15" / "UNSW-NB15 dataset" / "CSV Files" / "Training and Testing Sets" / "UNSW_NB15_testing-set.csv"),
            "--out_dir", str(ROOT / "Datasets" / "UNSW-NB15" / "outputs" / "baseline"),
        ],
    },
    "unsw_xgb_baseline": {
        "label": "UNSW-NB15 · XGB baseline",
        "description": "Run XGBoost baseline on official UNSW-NB15 train/test split.",
        "cmd": [
            PYTHON,
            "full_dataset_preprocessing/baseline_improved_UNSWNB15/train_xgb_unsw.py",
            "--train_path", str(ROOT / "Datasets" / "UNSW-NB15" / "UNSW-NB15 dataset" / "CSV Files" / "Training and Testing Sets" / "UNSW_NB15_training-set.csv"),
            "--test_path", str(ROOT / "Datasets" / "UNSW-NB15" / "UNSW-NB15 dataset" / "CSV Files" / "Training and Testing Sets" / "UNSW_NB15_testing-set.csv"),
            "--out_dir", str(ROOT / "Datasets" / "UNSW-NB15" / "outputs" / "baseline" / "xgb_baseline"),
            "--target_col", "label",
            "--seed", "42",
        ],
    },
    "unsw_rf_baseline": {
        "label": "UNSW-NB15 · RF baseline",
        "description": "Run Random Forest baseline on official UNSW-NB15 train/test split.",
        "cmd": [
            PYTHON,
            "full_dataset_preprocessing/baseline_improved_UNSWNB15/train_rf_unsw.py",
            "--train_path", str(ROOT / "Datasets" / "UNSW-NB15" / "UNSW-NB15 dataset" / "CSV Files" / "Training and Testing Sets" / "UNSW_NB15_training-set.csv"),
            "--test_path", str(ROOT / "Datasets" / "UNSW-NB15" / "UNSW-NB15 dataset" / "CSV Files" / "Training and Testing Sets" / "UNSW_NB15_testing-set.csv"),
            "--out_dir", str(ROOT / "Datasets" / "UNSW-NB15" / "outputs" / "baseline" / "rf_baseline"),
            "--target_col", "label",
            "--seed", "42",
        ],
    },
    "unsw_xgb_l1ao": {
        "label": "UNSW-NB15 · Leave-one-attack-out XGB",
        "description": "Run leave-one-attack-type-out evaluation with XGBoost.",
        "cmd": [
            PYTHON,
            "full_dataset_preprocessing/baseline_improved_UNSWNB15/leave_one_attack_type_out_xgb.py",
            "--train_path", str(ROOT / "Datasets" / "UNSW-NB15" / "UNSW-NB15 dataset" / "CSV Files" / "Training and Testing Sets" / "UNSW_NB15_training-set.csv"),
            "--test_path", str(ROOT / "Datasets" / "UNSW-NB15" / "UNSW-NB15 dataset" / "CSV Files" / "Training and Testing Sets" / "UNSW_NB15_testing-set.csv"),
            "--out_dir", str(ROOT / "Datasets" / "UNSW-NB15" / "outputs" / "leave_one_attack_type_out" / "xgb"),
            "--target_col", "label",
            "--seed", "42",
        ],
    },
    "unsw_rf_l1ao": {
        "label": "UNSW-NB15 · Leave-one-attack-out RF",
        "description": "Run leave-one-attack-type-out evaluation with Random Forest.",
        "cmd": [
            PYTHON,
            "full_dataset_preprocessing/baseline_improved_UNSWNB15/leave_one_attack_type_out_rf.py",
            "--train_path", str(ROOT / "Datasets" / "UNSW-NB15" / "UNSW-NB15 dataset" / "CSV Files" / "Training and Testing Sets" / "UNSW_NB15_training-set.csv"),
            "--test_path", str(ROOT / "Datasets" / "UNSW-NB15" / "UNSW-NB15 dataset" / "CSV Files" / "Training and Testing Sets" / "UNSW_NB15_testing-set.csv"),
            "--out_dir", str(ROOT / "Datasets" / "UNSW-NB15" / "outputs" / "leave_one_attack_type_out" / "rf"),
            "--target_col", "label",
            "--seed", "42",
        ],
    },
    "unsw_l1ao_analysis": {
        "label": "UNSW-NB15 · Analyze leave-one-attack-out",
        "description": "Summarize and compare RF/XGB leave-one-attack-type-out results.",
        "cmd": [
            PYTHON,
            "full_dataset_preprocessing/baseline_improved_UNSWNB15/analyze_leave_one_attack_type_out.py",
            "--rf_dir", str(ROOT / "Datasets" / "UNSW-NB15" / "outputs" / "leave_one_attack_type_out" / "rf"),
            "--xgb_dir", str(ROOT / "Datasets" / "UNSW-NB15" / "outputs" / "leave_one_attack_type_out" / "xgb"),
            "--out_dir", str(ROOT / "Datasets" / "UNSW-NB15" / "outputs" / "leave_one_attack_type_out" / "analysis"),
        ],
    },
    "unsw_l1ao_plots": {
        "label": "UNSW-NB15 · Leave-one-attack-out plots",
        "description": "Generate comparison plots from leave-one-attack-type-out analysis.",
        "cmd": [
            PYTHON,
            "full_dataset_preprocessing/baseline_improved_UNSWNB15/plot_leave_one_attack_type_results.py",
            "--comparison_csv", str(ROOT / "Datasets" / "UNSW-NB15" / "outputs" / "leave_one_attack_type_out" / "analysis" / "rf_vs_xgb_comparison.csv"),
            "--out_dir", str(ROOT / "Datasets" / "UNSW-NB15" / "outputs" / "leave_one_attack_type_out" / "analysis" / "plots"),
        ],
    },
}