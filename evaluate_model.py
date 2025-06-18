'''
- This module provides a utility class (`UtilMLEvals`) for evaluating both 
offline model predictions and in-training metrics for deep learning-based 
RUL models. It supports NumPy-based evaluation for post-inference metrics 
(e.g., RMSE, MAE, R², NASA score) and Keras-compatible metrics for model 
training (e.g., custom RMSE, R², scaled loss).


KEY FEATURES:
-------------
- Calculates standard evaluation metrics: RMSE, MAE, R²
- Implements NASA's exponential scoring function to penalize late predictions
- Includes custom Keras backend metrics for use in model.compile()
- Handles normalized and unnormalized RUL values via max_RUL scaling
- Designed for modular integration into ML/DL RUL pipelines

USE CASES:
----------
- Offline evaluation of predicted vs. true RUL values after training
- Live metric tracking in model.fit() using Keras backend functions
- Custom loss or scoring functions in deep RUL architectures

REQUIREMENTS:
-------------
- TensorFlow/Keras (for backend metrics)
- NumPy and Scikit-learn (for offline evaluation)


Aref Aasi, June 2024
'''


import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class UtilMLEvals:
    """
    Utility class for evaluating RUL prediction models.
    Supports both NumPy-based and Keras backend-based metrics.

    Args:
        max_RUL (float): Max cap for RUL normalization
    """

    def __init__(self, max_RUL: float = 130):
        self.max_RUL = max_RUL

    # =============================
    # NumPy-based Offline Evaluation
    # =============================

    def model_rmse(self, y_true, y_pred):
        """
        Compute RMSE between ground truth and predicted RUL.

        Args:
            y_true (np.ndarray): True RUL (uncapped, unnormalized)
            y_pred (np.ndarray): Predicted RUL (unnormalized)

        Returns:
            float: RMSE score
        """
        return np.sqrt(mean_squared_error(y_true, y_pred))

    def model_mae(self, y_true, y_pred):
        """
        Compute MAE between ground truth and predicted RUL.
        """
        return mean_absolute_error(y_true, y_pred)

    def model_r2(self, y_true, y_pred):
        """
        Compute R² score between true and predicted RUL.
        """
        return r2_score(y_true, y_pred)

    def model_score(self, y_true, y_pred):
        """
        NASA Exponential Scoring Function:
        Penalizes late predictions more than early ones.

        Args:
            y_true (np.ndarray): True RUL
            y_pred (np.ndarray): Predicted RUL

        Returns:
            float: Score (lower is better)
        """
        score = 0.0
        for i in range(len(y_true)):
            diff = y_pred[i] - y_true[i]
            if diff < 0:
                score += np.exp(-diff / 13) - 1
            else:
                score += np.exp(diff / 10) - 1
        return score

    # =============================
    # Keras-based Metrics (for training)
    # =============================

    def rmse_keras(self, y_true, y_pred):
        """
        RMSE metric for Keras (for normalized predictions).
        Scales by max_RUL to reverse normalization.
        """
        return K.sqrt(K.mean(K.square(self.max_RUL * (y_true - y_pred)), axis=-1))

    def r2_keras(self, y_true, y_pred):
        """
        R² (coefficient of determination) for Keras backend.
        """
        SS_res = K.sum(K.square(y_true - y_pred))
        SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
        return 1.0 - SS_res / (SS_tot + K.epsilon())

    def squared_error_scaled(self, y_true, y_pred):
        """
        Half-scaled squared error for Keras training loss.
        Good for Gaussian-style loss.
        
        
        """
        return K.square(self.max_RUL * (y_true - y_pred)) / 2
