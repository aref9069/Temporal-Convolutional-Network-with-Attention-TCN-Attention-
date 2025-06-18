'''

- This module provides visualization tools for comparing model performance on 
Remaining Useful Life (RUL) prediction tasks. It generates line plots and 
scatter plots to assess how closely the predicted RUL aligns with ground truth 
values across different models.


FUNCTION: `plot_result`
------------------------
Generates the following plots per model:
  1. Line plot showing predicted vs. true RUL for each engine.
  2. Scatter plot comparing predicted RUL to actual RUL, with a reference diagonal.

KEY FEATURES:
-------------
- Supports multiple models for visual comparison in a single figure.
- Displays RMSE and NASA scoring metrics in plot titles.
- Offers option to save figures to disk.
- Includes reference "ideal" line in scatter plots for visual calibration.

ARGS:
-----
- data_identifier (str): Dataset name or ID (e.g., 'FD001', 'FD002')
- ml_algorithms (list of str): Names of models (e.g., ['CNN', 'LSTM'])
- y_predicts (list of np.ndarray): Predicted RUL values from each model
- y_true (np.ndarray): Ground truth RUL values
- max_RUL (int): Maximum capped RUL used for plotting diagonal
- rmse_list (list of float): RMSE values per model
- score_list (list of float): NASA score values per model
- save_path (str, optional): Output path to save plot (e.g., './plots/FD001_results.png')


Aref Aasi, June 2024

'''

import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import os


def plot_result(
    data_identifier,
    ml_algorithms,
    y_predicts,
    y_true,
    max_RUL,
    rmse_list,
    score_list,
    save_path=None
):
    """
    Plot true vs. predicted RUL and scatter comparisons for multiple models.

    Args:
        data_identifier (str): Dataset ID (e.g., 'FD002')
        ml_algorithms (list): List of algorithm/model names
        y_predicts (list): List of prediction arrays (one per model)
        y_true (array): Ground truth RUL values
        max_RUL (int): Maximum RUL cap for plotting diagonal
        rmse_list (list): RMSE per model
        score_list (list): NASA score per model
        save_path (str, optional): Path to save the figure (e.g., './plots/FD002_results.png')
    """
    assert len(y_predicts) == len(ml_algorithms) == len(rmse_list) == len(score_list), \
        "Length mismatch among inputs"

    rows = len(ml_algorithms)
    fig = plt.figure(figsize=(15, 2.5 * rows))
    gs = gridspec.GridSpec(rows, 2, width_ratios=[5, 1])

    for i, algo in enumerate(ml_algorithms):
        y_pred = y_predicts[i]
        x_axis = np.arange(1, len(y_pred) + 1)

        # Line Plot: True vs Predicted RUL
        ax1 = plt.subplot(gs[2 * i])
        ax1.plot(x_axis, y_true, label="True", color='black')
        ax1.plot(x_axis, y_pred, label="Predicted", color='red')
        ax1.set_xlabel("Engine")
        ax1.set_ylabel("RUL")
        ax1.set_title(f"{algo} on {data_identifier}\nRMSE={rmse_list[i]:.2f}, Score={score_list[i]:.2e}")
        ax1.legend()
        ax1.grid(True)

        # Scatter Plot: Predicted vs True
        ax2 = plt.subplot(gs[2 * i + 1])
        ax2.scatter(y_true, y_pred, color='blue', alpha=0.6)
        diag = np.linspace(0, max_RUL, 100)
        ax2.plot(diag, diag, 'r--', label="Ideal")
        ax2.set_xlabel("True RUL")
        ax2.set_ylabel("Predicted RUL")
        ax2.set_title("Scatter")
        ax2.grid(True)
        ax2.axis("equal")

    fig.suptitle(f"Model Comparison on {data_identifier}", fontsize=14, y=1.02)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f" Plot saved to {save_path}")

    plt.show()
