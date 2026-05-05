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

COMMANDS.update({
    "feature_compare_raw": {
        "label": "Feature alignment - Compare raw features",
        "description": "Compare exported raw feature lists for IoT-23 and UNSW-NB15.",
        "cmd": [
            PYTHON,
            "feature_alignment/compare_features.py",
        ],
    },
    "feature_build_curated_alignment": {
        "label": "Feature alignment - Build curated alignment",
        "description": "Write the curated aligned-feature table used by the cross-dataset experiments.",
        "cmd": [
            PYTHON,
            "feature_alignment/build_curated_alignment.py",
        ],
    },
    "transfer_learning_core": {
        "label": "Transfer learning - Core run",
        "description": "Run the main cross-dataset transfer-learning experiment on the curated aligned subset.",
        "cmd": [
            PYTHON,
            "transfer_learning/run_transfer_learning.py",
        ],
    },
    "transfer_learning_updated_recipe": {
        "label": "Transfer learning - Updated recipe",
        "description": "Run the updated transfer-learning recipe with target calibration and threshold tuning.",
        "cmd": [
            PYTHON,
            "transfer_learning/transfer_learning_updated_recipe.py",
        ],
    },
    "transfer_learning_seed_stability": {
        "label": "Transfer learning - Seed stability",
        "description": "Run the updated transfer-learning recipe across multiple seeds and aggregate seed-stability outputs.",
        "cmd": [
            PYTHON,
            "transfer_learning/transfer_different_seed.py",
        ],
    },
    "transfer_learning_updated_plots": {
        "label": "Transfer learning - Updated recipe plots",
        "description": "Plot the updated transfer-learning recipe results.",
        "cmd": [
            PYTHON,
            "transfer_learning/plot_updated_recipe_results.py",
        ],
    },
    "transfer_learning_finalize": {
        "label": "Transfer learning - Final tables",
        "description": "Create thesis-ready transfer-learning tables and the verification report.",
        "cmd": [
            PYTHON,
            "transfer_learning/finalize_transfer_learning_results.py",
        ],
    },
    "transfer_hypothesis_create_pairs": {
        "label": "Transfer hypothesis - Create domain pairs",
        "description": "Generate pseudo-domain registries and cross-dataset pair manifests for the pairwise transfer hypothesis study.",
        "cmd": [
            PYTHON,
            "transfer_learning/hypothesis/create_domain_pairs.py",
        ],
    },
    "transfer_hypothesis_run_iot23_to_unsw": {
        "label": "Transfer hypothesis - Run IoT23 to UNSW",
        "description": "Run pairwise transfer-learning hypothesis experiments for the IoT-23 to UNSW-NB15 direction family.",
        "cmd": [
            PYTHON,
            "transfer_learning/hypothesis/run_pairwise_transfer_hypothesis.py",
            "--pair_family", "iot23_to_unsw",
            "--out_dir", str(ROOT / "transfer_learning" / "hypothesis" / "pairwise_runs_iot23_to_unsw_multi_seed"),
        ],
    },
    "transfer_hypothesis_run_unsw_to_iot23": {
        "label": "Transfer hypothesis - Run UNSW to IoT23",
        "description": "Run pairwise transfer-learning hypothesis experiments for the UNSW-NB15 to IoT-23 direction family.",
        "cmd": [
            PYTHON,
            "transfer_learning/hypothesis/run_pairwise_transfer_hypothesis.py",
            "--pair_family", "unsw_to_iot23",
            "--out_dir", str(ROOT / "transfer_learning" / "hypothesis" / "pairwise_runs_unsw_to_iot23_multi_seed"),
        ],
    },
    "transfer_hypothesis_analyze_iot23_to_unsw": {
        "label": "Transfer hypothesis - Analyze IoT23 to UNSW",
        "description": "Analyze pairwise hypothesis runs for the IoT-23 to UNSW-NB15 direction family.",
        "cmd": [
            PYTHON,
            "transfer_learning/hypothesis/analyze_pairwise_transfer_hypothesis.py",
            "--run_dir", str(ROOT / "transfer_learning" / "hypothesis" / "pairwise_runs_iot23_to_unsw_multi_seed"),
        ],
    },
    "transfer_hypothesis_analyze_unsw_to_iot23": {
        "label": "Transfer hypothesis - Analyze UNSW to IoT23",
        "description": "Analyze pairwise hypothesis runs for the UNSW-NB15 to IoT-23 direction family.",
        "cmd": [
            PYTHON,
            "transfer_learning/hypothesis/analyze_pairwise_transfer_hypothesis.py",
            "--run_dir", str(ROOT / "transfer_learning" / "hypothesis" / "pairwise_runs_unsw_to_iot23_multi_seed"),
        ],
    },
    "transfer_hypothesis_compare_directions": {
        "label": "Transfer hypothesis - Compare directions",
        "description": "Combine both pairwise hypothesis analysis folders into a cross-direction comparison.",
        "cmd": [
            PYTHON,
            "transfer_learning/hypothesis/compare_pairwise_transfer_directions.py",
        ],
    },
    "transfer_hypothesis_plot_directions": {
        "label": "Transfer hypothesis - Plot combined directions",
        "description": "Plot the combined direction-analysis figures for the pairwise transfer hypothesis study.",
        "cmd": [
            PYTHON,
            "transfer_learning/hypothesis/plot_combined_direction_analysis.py",
        ],
    },
    "cross_domain_shift_run": {
        "label": "Cross-domain shift - Run experiment",
        "description": "Run the curated cross-domain shift experiment on IoT-23 and UNSW-NB15.",
        "cmd": [
            PYTHON,
            "cross_domain_shift/run_cross_domain_shift.py",
        ],
    },
    "cross_domain_shift_progression_plot": {
        "label": "Cross-domain shift - Progression plot",
        "description": "Build the staged progression plot across the cross-domain shift run folders.",
        "cmd": [
            PYTHON,
            "cross_domain_shift/plot_experiment_progression.py",
        ],
    },
    "cross_domain_shift_feature_importance_analysis": {
        "label": "Cross-domain shift - Feature importance analysis",
        "description": "Analyze cross-domain shift feature-importance patterns across the staged outputs folders.",
        "cmd": [
            PYTHON,
            "cross_domain_shift/analyze_cross_domain_feature_importance.py",
        ],
    },
    "early_in_domain_iot23_rf": {
        "label": "Early detection - IoT23 in-domain RF",
        "description": "Run in-domain early detection on IoT-23 with Random Forest.",
        "cmd": [
            PYTHON,
            "early_detection/in_domain_early_detection/run_iot23_in_domain_early_detection.py",
        ],
    },
    "early_in_domain_unsw_rf": {
        "label": "Early detection - UNSW in-domain RF",
        "description": "Run in-domain early detection on UNSW-NB15 with Random Forest.",
        "cmd": [
            PYTHON,
            "early_detection/in_domain_early_detection/run_unsw_in_domain_early_detection.py",
        ],
    },
    "early_in_domain_iot23_mlp": {
        "label": "Early detection - IoT23 in-domain MLP",
        "description": "Run in-domain early detection on IoT-23 with the shared MLP runner.",
        "cmd": [
            PYTHON,
            "early_detection/in_domain_early_detection/run_in_domain_mlp_early_detection.py",
            "--dataset", "iot23",
            "--out_dir", str(ROOT / "early_detection" / "in_domain_early_detection" / "outputs_iot23_mlp_exp1"),
        ],
    },
    "early_in_domain_unsw_mlp": {
        "label": "Early detection - UNSW in-domain MLP",
        "description": "Run in-domain early detection on UNSW-NB15 with the shared MLP runner.",
        "cmd": [
            PYTHON,
            "early_detection/in_domain_early_detection/run_in_domain_mlp_early_detection.py",
            "--dataset", "unsw",
            "--out_dir", str(ROOT / "early_detection" / "in_domain_early_detection" / "outputs_unsw_mlp_exp1"),
        ],
    },
    "early_in_domain_analyze_runs": {
        "label": "Early detection - Analyze in-domain runs",
        "description": "Aggregate and compare the in-domain early-detection run folders.",
        "cmd": [
            PYTHON,
            "early_detection/in_domain_early_detection/analyze_in_domain_runs.py",
        ],
    },
    "early_in_domain_analyze_multiseed": {
        "label": "Early detection - Analyze in-domain multi-seed",
        "description": "Aggregate repeated in-domain early-detection seed runs.",
        "cmd": [
            PYTHON,
            "early_detection/in_domain_early_detection/analyze_in_domain_multi_seed.py",
        ],
    },
    "early_cross_domain_rf": {
        "label": "Early detection - Cross-domain RF",
        "description": "Run cross-domain early detection with Random Forest on the curated aligned subset.",
        "cmd": [
            PYTHON,
            "early_detection/cross_domain_early_detection/run_cross_domain_early_detection.py",
        ],
    },
    "early_cross_domain_mlp": {
        "label": "Early detection - Cross-domain MLP",
        "description": "Run cross-domain early detection with MLP on the curated aligned subset.",
        "cmd": [
            PYTHON,
            "early_detection/cross_domain_early_detection/run_cross_domain_mlp_early_detection.py",
        ],
    },
    "early_cross_domain_analyze_runs": {
        "label": "Early detection - Analyze cross-domain runs",
        "description": "Aggregate and compare the cross-domain early-detection runs.",
        "cmd": [
            PYTHON,
            "early_detection/cross_domain_early_detection/analyze_cross_domain_runs.py",
        ],
    },
    "early_cross_domain_multiseed_run": {
        "label": "Early detection - Cross-domain multi-seed",
        "description": "Run the matched cross-domain early-detection RF and MLP configurations across multiple seeds.",
        "cmd": [
            PYTHON,
            "early_detection/cross_domain_early_detection/run_cross_domain_multi_seed.py",
        ],
    },
    "early_cross_domain_multiseed_analyze": {
        "label": "Early detection - Analyze cross-domain multi-seed",
        "description": "Aggregate repeated cross-domain early-detection seed runs.",
        "cmd": [
            PYTHON,
            "early_detection/cross_domain_early_detection/analyze_cross_domain_multi_seed.py",
        ],
    },
    "early_cross_domain_sensitivity_run": {
        "label": "Early detection - Cross-domain sensitivity",
        "description": "Run the cross-domain size-sensitivity and eval-cap sensitivity studies.",
        "cmd": [
            PYTHON,
            "early_detection/cross_domain_early_detection/run_cross_domain_sensitivity.py",
        ],
    },
    "early_cross_domain_sensitivity_analyze": {
        "label": "Early detection - Analyze cross-domain sensitivity",
        "description": "Analyze the cross-domain early-detection sensitivity experiments against the repeated baseline.",
        "cmd": [
            PYTHON,
            "early_detection/cross_domain_early_detection/analyze_cross_domain_sensitivity.py",
        ],
    },
    "early_transfer_learning_single_iot23_to_unsw": {
        "label": "Early detection - Transfer run IoT23 to UNSW",
        "description": "Run one matched transfer-learning early-detection experiment for IoT-23 to UNSW-NB15.",
        "cmd": [
            PYTHON,
            "early_detection/transfer_learning_based_early_detection/run_transfer_learning_early_detection.py",
            "--direction", "iot23_to_unsw",
            "--target_train_rows", "5000",
            "--out_dir", str(ROOT / "early_detection" / "transfer_learning_based_early_detection" / "outputs_iot23_to_unsw_budget5k_seed42"),
        ],
    },
    "early_transfer_learning_single_unsw_to_iot23": {
        "label": "Early detection - Transfer run UNSW to IoT23",
        "description": "Run one matched transfer-learning early-detection experiment for UNSW-NB15 to IoT-23.",
        "cmd": [
            PYTHON,
            "early_detection/transfer_learning_based_early_detection/run_transfer_learning_early_detection.py",
            "--direction", "unsw_to_iot23",
            "--target_train_rows", "5000",
            "--out_dir", str(ROOT / "early_detection" / "transfer_learning_based_early_detection" / "outputs_unsw_to_iot23_budget5k_seed42"),
        ],
    },
    "early_transfer_learning_multiseed_run": {
        "label": "Early detection - Transfer multi-seed",
        "description": "Run the transfer-learning early-detection matrix across directions and target-train budgets.",
        "cmd": [
            PYTHON,
            "early_detection/transfer_learning_based_early_detection/run_transfer_learning_multi_seed.py",
        ],
    },
    "early_transfer_learning_multiseed_analyze": {
        "label": "Early detection - Analyze transfer multi-seed",
        "description": "Aggregate repeated transfer-learning early-detection runs and summarize transfer gains.",
        "cmd": [
            PYTHON,
            "early_detection/transfer_learning_based_early_detection/analyze_transfer_learning_multi_seed.py",
        ],
    },
})

COMMANDS.update({
    "feature_export_iot23_raw": {
        "label": "Feature alignment - Export IoT23 raw features",
        "description": "Export raw IoT-23 feature names from the processed training parquet used for alignment work.",
        "cmd": [
            PYTHON,
            "data_preprocessing/export_iot23_feature_space.py",
            "--train_path", str(ROOT / "Datasets" / "IoT23" / "processed_test_sample" / "iot23" / "train.parquet"),
            "--out_dir", str(ROOT / "feature_alignment"),
            "--out_name", "iot23_features.json",
        ],
    },
    "feature_export_iot23_model": {
        "label": "Feature alignment - Export IoT23 model features",
        "description": "Export transformed IoT-23 model feature names from the trained Random Forest pipeline.",
        "cmd": [
            PYTHON,
            "data_preprocessing/iot23_export_model_feature_space.py",
            "--model_path", str(ROOT / "Datasets" / "IoT23" / "processed_test_sample" / "iot23" / "rf_baseline" / "rf_model.joblib"),
            "--out_dir", str(ROOT / "feature_alignment"),
            "--out_name", "iot23_model_features.json",
        ],
    },
    "feature_export_unsw_raw": {
        "label": "Feature alignment - Export UNSW raw features",
        "description": "Export raw UNSW-NB15 feature names from the official train split used for alignment work.",
        "cmd": [
            PYTHON,
            "data_preprocessing/export_unsw_feature_space.py",
            "--train_path", str(ROOT / "Datasets" / "UNSW-NB15" / "UNSW-NB15 dataset" / "CSV Files" / "Training and Testing Sets" / "UNSW_NB15_training-set.csv"),
            "--out_dir", str(ROOT / "feature_alignment"),
            "--out_name", "unsw_features.json",
        ],
    },
    "iot_loso_xgb": {
        "label": "IoT-23 - LOSO XGB",
        "description": "Run Leave-One-Scenario-Out evaluation with XGBoost on IoT-23.",
        "cmd": [
            PYTHON,
            "data_preprocessing/loso_xgboost.py",
            "--data_file", str(ROOT / "Datasets" / "IoT23" / "processed_test_sample" / "iot23" / "all_flows.parquet"),
            "--out_dir", str(ROOT / "Datasets" / "IoT23" / "processed_test_sample" / "iot23" / "xgb_loso"),
            "--target_col", "label_binary",
            "--seed", "42",
        ],
    },
    "iot_loso_rf_analysis": {
        "label": "IoT-23 - Analyze LOSO RF",
        "description": "Analyze IoT-23 Random Forest LOSO fold summaries and generate ranked views.",
        "cmd": [
            PYTHON,
            "data_preprocessing/analyze_loso_results_rf.py",
            "--loso_dir", str(ROOT / "Datasets" / "IoT23" / "processed_test_sample" / "iot23" / "rf_loso"),
            "--out_dir", str(ROOT / "Datasets" / "IoT23" / "processed_test_sample" / "iot23" / "rf_loso" / "analysis"),
        ],
    },
    "iot_model_comparison": {
        "label": "IoT-23 - Compare RF and XGB",
        "description": "Compare pooled and LOSO summary metrics between the IoT-23 Random Forest and XGBoost runs.",
        "cmd": [
            PYTHON,
            "data_preprocessing/compare_baseline_models.py",
        ],
    },
    "iot_feature_transfer_analysis": {
        "label": "IoT-23 - Analyze feature transfer",
        "description": "Aggregate fold-wise feature importances from RF and XGB LOSO runs for transferability analysis.",
        "cmd": [
            PYTHON,
            "data_preprocessing/analyze_feature_transfer.py",
            "--rf_dir", str(ROOT / "Datasets" / "IoT23" / "processed_test_sample" / "iot23" / "rf_loso"),
            "--xgb_dir", str(ROOT / "Datasets" / "IoT23" / "processed_test_sample" / "iot23" / "xgb_loso"),
            "--out_dir", str(ROOT / "Datasets" / "IoT23" / "processed_test_sample" / "iot23" / "feature_transfer_analysis"),
        ],
    },
    "iot_feature_stability_compute": {
        "label": "IoT-23 - Compute feature stability",
        "description": "Compute stability and transfer-utility summaries from the IoT-23 LOSO feature-importance exports.",
        "cmd": [
            PYTHON,
            "data_preprocessing/compute_feature_stability.py",
            "--rf_long", str(ROOT / "Datasets" / "IoT23" / "processed_test_sample" / "iot23" / "feature_transfer_analysis" / "rf_fold_feature_importances_long.csv"),
            "--xgb_long", str(ROOT / "Datasets" / "IoT23" / "processed_test_sample" / "iot23" / "feature_transfer_analysis" / "xgb_fold_feature_importances_long.csv"),
            "--out_dir", str(ROOT / "Datasets" / "IoT23" / "processed_test_sample" / "iot23" / "feature_stability_analysis"),
        ],
    },
    "research_suite_align": {
        "label": "Research suite - Alignment inspection",
        "description": "Run the combined research analysis suite alignment subcommand on the full processed CSV exports.",
        "cmd": [
            PYTHON,
            "data_preprocessing/research_analysis_suite.py",
            "align",
            "--csv_a", str(ROOT / "Datasets" / "IoT23" / "processed_full" / "iot23" / "iot23_final.csv"),
            "--csv_b", str(ROOT / "Datasets" / "UNSW_NB15" / "processed_full" / "unsw_nb15" / "unsw_final.csv"),
            "--label_col", "label",
            "--drop_cols", "timestamp", "scenario_id",
            "--out_dir", str(ROOT / "Datasets" / "cross_dataset_eval" / "alignment_suite"),
        ],
    },
    "research_suite_stability_plots": {
        "label": "Research suite - Stability plots",
        "description": "Run the combined research analysis suite feature-stability plotting subcommand.",
        "cmd": [
            PYTHON,
            "data_preprocessing/research_analysis_suite.py",
            "stability_plots",
            "--rf_summary", str(ROOT / "Datasets" / "IoT23" / "processed_test_sample" / "iot23" / "feature_stability_analysis" / "rf_feature_stability_summary.csv"),
            "--xgb_summary", str(ROOT / "Datasets" / "IoT23" / "processed_test_sample" / "iot23" / "feature_stability_analysis" / "xgb_feature_stability_summary.csv"),
            "--out_dir", str(ROOT / "Datasets" / "IoT23" / "processed_test_sample" / "iot23" / "feature_stability_plots_suite"),
        ],
    },
    "research_suite_cross_eval": {
        "label": "Research suite - Cross-dataset eval",
        "description": "Run the combined research analysis suite cross-dataset evaluation subcommand.",
        "cmd": [
            PYTHON,
            "data_preprocessing/research_analysis_suite.py",
            "cross_eval",
            "--iot_csv", str(ROOT / "Datasets" / "IoT23" / "processed_full" / "iot23" / "iot23_final.csv"),
            "--unsw_csv", str(ROOT / "Datasets" / "UNSW_NB15" / "processed_full" / "unsw_nb15" / "unsw_final.csv"),
            "--out_dir", str(ROOT / "Datasets" / "cross_dataset_eval" / "suite_outputs"),
            "--models", "rf", "xgb",
            "--drop_cols", "timestamp", "scenario_id",
        ],
    },
    "research_suite_cross_plots": {
        "label": "Research suite - Cross-dataset plots",
        "description": "Run the combined research analysis suite plotting subcommand for cross-dataset results.",
        "cmd": [
            PYTHON,
            "data_preprocessing/research_analysis_suite.py",
            "cross_plots",
            "--summary_csv", str(ROOT / "Datasets" / "cross_dataset_eval" / "suite_outputs" / "cross_dataset_summary.csv"),
            "--out_dir", str(ROOT / "Datasets" / "cross_dataset_eval" / "suite_outputs" / "plots"),
        ],
    },
    "iot_dataset_statistics": {
        "label": "IoT-23 - Dataset statistics",
        "description": "Generate thesis-ready dataset statistics and plots from the processed IoT-23 full dataset export.",
        "cmd": [
            PYTHON,
            "data_preprocessing/dataset_statistics_iot23.py",
            "--processed_dir", str(ROOT / "Datasets" / "IoT23" / "processed_full"),
            "--dataset_name", "iot23",
        ],
    },
    "iot_export_full_splits": {
        "label": "IoT-23 - Export full train/val/test splits",
        "description": "Stream the per-scenario IoT-23 parquet files into the consolidated full train, val, and test splits.",
        "cmd": [
            PYTHON,
            "data_preprocessing/export_full_iot23_splits_from_scenarios.py",
            "--scenario_dir", str(ROOT / "Datasets" / "IoT23" / "processed_full" / "iot23" / "processed_scenarios"),
            "--out_dir", str(ROOT / "Datasets" / "IoT23" / "processed_full" / "iot23"),
        ],
    },
    "full_iot23_dataset_statistics": {
        "label": "Full IoT-23 - Scenario statistics",
        "description": "Compute scenario-level IoT-23 dataset statistics directly from the processed per-scenario parquet files.",
        "cmd": [
            PYTHON,
            "full_dataset_preprocessing/full_iot23_dataset_statistics.py",
            "--input-dir", str(ROOT / "Datasets" / "IoT23" / "processed_full" / "iot23" / "processed_scenarios"),
            "--output-dir", str(ROOT / "Datasets" / "IoT23" / "processed_full" / "iot23" / "full_dataset_statistics"),
            "--top-features-heatmap", "20",
        ],
    },
    "iot23_rf_baseline_improved": {
        "label": "Full IoT-23 - Improved RF baseline",
        "description": "Run the memory-safer improved Random Forest baseline on the full IoT-23 train, val, and test splits.",
        "cmd": [
            PYTHON,
            "full_dataset_preprocessing/baseline_improved_IoT23/train_baseline_rf_improved.py",
            "--data_dir", str(ROOT / "Datasets" / "IoT23" / "processed_full" / "iot23"),
            "--out_dir", str(ROOT / "Datasets" / "IoT23" / "processed_full" / "iot23" / "rf_baseline_improved"),
            "--target_col", "label_binary",
            "--seed", "42",
        ],
    },
    "iot23_xgb_baseline_improved": {
        "label": "Full IoT-23 - Improved XGB baseline",
        "description": "Run the memory-safer improved XGBoost baseline on the full IoT-23 train, val, and test splits.",
        "cmd": [
            PYTHON,
            "full_dataset_preprocessing/baseline_improved_IoT23/train_baseline_xgboost_improved.py",
            "--data_dir", str(ROOT / "Datasets" / "IoT23" / "processed_full" / "iot23"),
            "--out_dir", str(ROOT / "Datasets" / "IoT23" / "processed_full" / "iot23" / "xgb_baseline_improved"),
            "--target_col", "label_binary",
            "--seed", "42",
        ],
    },
    "iot23_xgb_baseline_threshold_improved": {
        "label": "Full IoT-23 - Improved XGB threshold baseline",
        "description": "Run the improved XGBoost baseline with an explicit probability threshold on the full IoT-23 splits.",
        "cmd": [
            PYTHON,
            "full_dataset_preprocessing/baseline_improved_IoT23/train_baseline_xgboost_improved_threshold.py",
            "--data_dir", str(ROOT / "Datasets" / "IoT23" / "processed_full" / "iot23"),
            "--out_dir", str(ROOT / "Datasets" / "IoT23" / "processed_full" / "iot23" / "xgb_baseline_threshold_improved"),
            "--target_col", "label_binary",
            "--seed", "42",
            "--decision_threshold", "0.50",
        ],
    },
    "iot23_rf_full_loso": {
        "label": "Full IoT-23 - RF LOSO",
        "description": "Run scenario-level full-dataset IoT-23 LOSO evaluation with the improved Random Forest pipeline.",
        "cmd": [
            PYTHON,
            "full_dataset_preprocessing/baseline_improved_IoT23/iot23_rf_full_LOSO.py",
            "--scenario_dir", str(ROOT / "Datasets" / "IoT23" / "processed_full" / "iot23" / "processed_scenarios"),
            "--out_dir", str(ROOT / "Datasets" / "IoT23" / "processed_full" / "iot23" / "rf_full_loso"),
            "--target_col", "label_binary",
            "--seed", "42",
        ],
    },
    "iot23_xgb_full_loso": {
        "label": "Full IoT-23 - XGB LOSO",
        "description": "Run scenario-level full-dataset IoT-23 LOSO evaluation with the improved XGBoost pipeline.",
        "cmd": [
            PYTHON,
            "full_dataset_preprocessing/baseline_improved_IoT23/iot23_xgbost_full_LOSO.PY",
            "--scenario_dir", str(ROOT / "Datasets" / "IoT23" / "processed_full" / "iot23" / "processed_scenarios"),
            "--out_dir", str(ROOT / "Datasets" / "IoT23" / "processed_full" / "iot23" / "xgb_full_loso"),
            "--target_col", "label_binary",
            "--seed", "42",
            "--decision_threshold", "0.50",
        ],
    },
    "unsw_feature_stability_transfer": {
        "label": "UNSW-NB15 - Feature stability transfer",
        "description": "Analyze fold-wise UNSW leave-one-attack-type-out feature importances for cross-model stability and transferability.",
        "cmd": [
            PYTHON,
            "full_dataset_preprocessing/baseline_improved_UNSWNB15/analyze_feature_stability_transfer.py",
            "--rf_dir", str(ROOT / "Datasets" / "UNSW-NB15" / "outputs" / "leave_one_attack_type_out" / "rf"),
            "--xgb_dir", str(ROOT / "Datasets" / "UNSW-NB15" / "outputs" / "leave_one_attack_type_out" / "xgb"),
            "--out_dir", str(ROOT / "Datasets" / "UNSW-NB15" / "outputs" / "leave_one_attack_type_out" / "feature_stability_analysis"),
        ],
    },
    "transfer_learning_threshold_target_only_unsw_to_iot23_0p05": {
        "label": "Transfer learning - Threshold analysis target-only 0.05",
        "description": "Analyze threshold sensitivity for the updated UNSW-NB15 to IoT-23 target-only 5 percent run.",
        "cmd": [
            PYTHON,
            "transfer_learning/analyze_thresholds.py",
            "--predictions_csv", str(ROOT / "transfer_learning" / "outputs_updated_recipe" / "unsw_train_to_iot23_test" / "target_only_updated_frac_0p05" / "predictions.csv"),
            "--out_dir", str(ROOT / "transfer_learning" / "threshold_analysis" / "unsw_to_iot23_target_only_0p05"),
        ],
    },
    "transfer_learning_threshold_transfer_unsw_to_iot23_0p05": {
        "label": "Transfer learning - Threshold analysis transfer 0.05",
        "description": "Analyze threshold sensitivity for the updated UNSW-NB15 to IoT-23 transfer-learning 5 percent run.",
        "cmd": [
            PYTHON,
            "transfer_learning/analyze_thresholds.py",
            "--predictions_csv", str(ROOT / "transfer_learning" / "outputs_updated_recipe" / "unsw_train_to_iot23_test" / "transfer_learning_updated_frac_0p05" / "predictions.csv"),
            "--out_dir", str(ROOT / "transfer_learning" / "threshold_analysis" / "unsw_to_iot23_transfer_learning_0p05"),
        ],
    },
    "transfer_learning_more_plots": {
        "label": "Transfer learning - Extra figures",
        "description": "Generate the extra transfer-learning seed, gain, threshold, and feature-shift figures.",
        "cmd": [
            PYTHON,
            "transfer_learning/more_plots.py",
        ],
    },
})
