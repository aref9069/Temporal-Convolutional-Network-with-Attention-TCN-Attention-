'''

==================================================================================
 Main Pipeline Script for RUL Estimation Using NASA C-MAPSS Dataset
==================================================================================

DESCRIPTION:
------------
This script serves as the main driver for training, evaluating, and comparing
machine learning and deep learning models on the NASA C-MAPSS datasets for
Remaining Useful Life (RUL) prediction.

It orchestrates the full pipeline including:
  - Data loading and cleaning
  - Feature engineering and preprocessing
  - Model training and prediction
  - Performance evaluation (RMSE, NASA Score)
  - Optional visualization and result saving

FEATURES:
---------
- Modular architecture supporting multiple models (e.g., BLSTM-CNN, LSTM, MLP)
- Configurable hyperparameters via `data_ml_settings` dictionary
- Support for plotting predicted vs. true RUL curves and scatter plots
- Saves results (RMSE & Score) to CSV files in `./result/` directory

USAGE:
------
Run from command line:

    python main.py <path_to_dataset_dir> <show_plots: true|false>

Example:

    python main.py ./CMAPSSData true

CONFIGURABLE COMPONENTS:
-------------------------
- `data_identifiers`: List of dataset subsets (e.g., 'FD001', 'FD002', ...)
- `ml_algorithms`: List of models to evaluate (must be implemented and imported)
- `data_ml_settings`: Dictionary controlling max RUL, sequence length, batch size, epochs, and learning rate

MODULE DEPENDENCIES:
--------------------
- `data_cleanup.py`: Loads and parses raw C-MAPSS files
- `features_selection.py`: Applies normalization, cumulative features, and sequence generation
- `evaluate_model.py`: Contains evaluation metrics (RMSE, MAE, R², NASA score)
- `plots.py`: (conditionally imported) For visualization of prediction results
- Model files: Each model (e.g., `blstm_cnn.py`) must implement `do_rul_estimation_with_<model>()`

OUTPUT:
-------
- RMSE and Score tables for each model-dataset combination
- Optional plots saved under `./result/plots/`

AUTHOR:
-------
Aref Aasi, August 2024

'''

import os
import sys
import argparse
from datetime import datetime
import pandas as pd
import numpy as np
import pathlib
import tensorflow as tf

# Local utility imports
import data_cleanup, features_selection
from evaluate_model import UtilMLEvals
import common, preprocessing
import blstm_cnn  # Add other models as needed

# ----------------------------
# Argument Parsing
# ----------------------------
parser = argparse.ArgumentParser(description="Run RUL estimation pipeline on C-MAPSS datasets.")
parser.add_argument("data_dir", type=str, help="Path to the C-MAPSS dataset folder.")
parser.add_argument("show_plots", type=str, choices=["true", "false"], help="Whether to display result plots.")
args = parser.parse_args()

data_dir = args.data_dir
show_plots = args.show_plots.lower() == "true"

if show_plots:
    import matplotlib.pyplot as plt
    import plots

# ----------------------------
# Directory Setup
# ----------------------------
result_dir = "./result"
plot_dir = os.path.join(result_dir, "plots")
pathlib.Path(result_dir).mkdir(parents=True, exist_ok=True)
pathlib.Path(plot_dir).mkdir(parents=True, exist_ok=True)

# ----------------------------
# Configurations
# ----------------------------
data_identifiers = ['FD001']  # Add more: 'FD002', 'FD003', 'FD004'
ml_algorithms = ['BLSTM_CNN']  # Add: 'CNN', 'LSTM', 'MLP', 'SVR', 'RVR'

data_ml_settings = {
    'max_RUL': 130,
    'sequence_length': 15,
    'batch_size': 25,
    'epochs': 10,
    'alpha': 1e-3,
}

# ----------------------------
# Model Dispatcher
# ----------------------------
def get_model_by_name(name):
    return {
        'BLSTM_CNN': blstm_cnn.do_rul_estimation_with_blstm_cnn,
        # 'CNN': cnn.do_rul_estimation_with_cnn,
        # 'LSTM': lstm.do_rul_estimation_with_lstm,
        # 'MLP': mlp.do_rul_estimation_with_mlp,
        # 'SVR': svr.do_rul_estimation_with_svr,
        # 'RVR': rvr.do_rul_estimation_with_rvr,
    }.get(name)

# ----------------------------
# Main Workflow
# ----------------------------
def main():
    RMSE_df = pd.DataFrame(index=ml_algorithms, columns=data_identifiers)
    SCORE_df = pd.DataFrame(index=ml_algorithms, columns=data_identifiers)
    ml_evals = UtilMLEvals(data_ml_settings['max_RUL'])

    for dataset in data_identifiers:
        print(f"\n=== Processing Dataset: {dataset} ===")

        # A. Load and preprocess data
        train_df, test_df, truth_df = data_cleanup.ingest_and_cleanup_data(data_dir, dataset)
        train_x, train_y, test_x, test_y = features_selection.preprocess_data(train_df, test_df, truth_df, data_ml_settings)

        print(f"✔ Data shapes — Train: {train_x.shape}, Test: {test_x.shape}")

        # B. Ground truth for test set
        rul_path = os.path.join(data_dir, f"RUL_{dataset}.txt")
        y_true = pd.read_csv(rul_path, sep=r'\s+', header=None).dropna(axis=1).iloc[:, 0].values

        # C. Run all selected models
        y_predicts = []
        for algo in ml_algorithms:
            print(f"\n→ Running model: {algo}")
            start = datetime.now()

            model_func = get_model_by_name(algo)
            if model_func is None:
                print(f"[⚠️] Model '{algo}' not implemented.")
                continue

            try:
                y_pred_norm = model_func(dataset, train_x, train_y, test_x, test_y, data_ml_settings)
                y_pred = y_pred_norm * data_ml_settings['max_RUL']
                y_predicts.append(y_pred)

                rmse = ml_evals.model_rmse(y_true, y_pred)
                score = ml_evals.model_score(y_true, y_pred)

                RMSE_df.loc[algo, dataset] = f"{rmse:.2f}"
                SCORE_df.loc[algo, dataset] = f"{score:.2e}"

                print(f"✓ {algo} completed in {datetime.now() - start} | RMSE: {rmse:.2f} | Score: {score:.2e}")

            except Exception as e:
                print(f"[❌ Error in {algo}] {str(e)}")

        # D. Plot results
        if show_plots and y_predicts:
            try:
                plots.plot_result(
                    data_identifier=dataset,
                    ml_algorithms=ml_algorithms,
                    y_predicts=y_predicts,
                    y_true=y_true,
                    max_RUL=data_ml_settings['max_RUL'],
                    rmse_list=RMSE_df[dataset].values,
                    score_list=SCORE_df[dataset].values
                )
                plt.savefig(os.path.join(plot_dir, f"{dataset}_{'_'.join(ml_algorithms)}.png"))
            except Exception as e:
                print(f"[ Plot Error] {str(e)}")

        print(f"=== Finished: {dataset} ===\n{'='*60}")

    # E. Save and summarize results
    RMSE_df.to_csv(os.path.join(result_dir, 'RMSE.csv'))
    SCORE_df.to_csv(os.path.join(result_dir, 'SCORE.csv'))

    print("\n Final RMSE Comparison:\n", RMSE_df)
    print("\n Final Score Comparison:\n", SCORE_df)

    if show_plots:
        plt.show()

# ----------------------------
# Entry Point
# ----------------------------
if __name__ == "__main__":
    main()
