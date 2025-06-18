'''
===============================================================================
 CNN + BiLSTM + Attention Model for RUL Prediction
===============================================================================

- This module defines a deep learning architecture that combines:
  - 1D Convolutional Neural Networks (CNNs) for local feature extraction,
  - Bidirectional Long Short-Term Memory (BLSTM) layers for capturing temporal dependencies,
  - An attention mechanism to focus on relevant time steps in the sequence.


- The model is tailored for Remaining Useful Life (RUL) prediction using the
NASA C-MAPSS dataset. It processes sequential sensor data and predicts the 
remaining time before failure for each engine unit.

MAIN FUNCTION:
--------------
- `do_rul_estimation_with_blstm_cnn(...)`: Trains and evaluates the model using
   training/test features and labels. Returns predicted RULs for test set.

KEY COMPONENTS:
---------------
- Conv1D: Extracts local temporal features
- AveragePooling1D: Reduces sequence length to prevent overfitting
- BiLSTM: Captures bidirectional temporal patterns
- Attention Layer: Dynamically weights time steps to enhance model interpretability
- GlobalAveragePooling1D: Aggregates final sequence embeddings
- Dense: Outputs a single RUL estimate per sequence

CONFIGURATION:
--------------
- Accepts training/testing data in (samples, time steps, features) format
- Hyperparameters like learning rate, batch size, and max RUL are passed via `data_ml_settings`
- Supports validation split and early stopping

EVALUATION:
-----------
- Returns RMSE, MAE, R², and NASA score metrics
- Compatible with normalized or capped RUL targets

EXAMPLE USAGE:
--------------
    y_pred = do_rul_estimation_with_blstm_cnn(
        data_identifier='FD002',
        train_PHM_x=train_x,
        train_PHM_y=train_y,
        test_PHM_x=test_x,
        test_PHM_y=test_y,
        data_ml_settings=config
    )

AUTHOR:
-------
Aref Aasi, July 2024

'''

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Conv2D, AveragePooling2D, Dropout, Bidirectional, LSTM,
    Multiply, TimeDistributed, Reshape, GlobalAveragePooling1D, Flatten
)
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import numpy as np
import sys
sys.path.append("..")

from evaluate_model import UtilMLEvals


def attention_layer(inputs):
    """
    Temporal attention layer.
    Args:
        inputs: 3D tensor (batch, time steps, features)
    Returns:
        attention-weighted output
    """
    attention = Dense(1, activation='tanh')(inputs)  # (batch, time, 1)
    attention = tf.keras.layers.Softmax(axis=1)(attention)  # across time
    attended = tf.keras.layers.Multiply()([inputs, attention])  # (batch, time, features)
    return attended


def do_rul_estimation_with_blstm_cnn(data_identifier, train_PHM_x, train_PHM_y, test_PHM_x, test_PHM_y, data_ml_settings):
    """
    Train and evaluate a CNN + BLSTM + Attention model for RUL prediction.

    Args:
        data_identifier (str): Dataset name (e.g. 'FD002')
        train_PHM_x (np.array): Training features (samples, seq_len, features)
        train_PHM_y (np.array): Training labels
        test_PHM_x (np.array): Test features
        test_PHM_y (np.array): Test labels
        data_ml_settings (dict): Configuration settings

    Returns:
        np.array: Flattened prediction vector
    """
    ml_evals_util = UtilMLEvals(data_ml_settings['max_RUL'])
    alpha = data_ml_settings['alpha']
    batch_size = data_ml_settings['batch_size']
    epochs = data_ml_settings['epochs']

    print('Begin training BLSTM-CNN model...')

    # === Input shapes ===
    input_shape = train_PHM_x.shape[1:]  # (sequence_length, num_features)
    output_dim = 1  # Scalar RUL prediction

    # === Model Architecture ===
    input_layer = Input(shape=input_shape)  # (T, F)

    # Conv1D path
    x = tf.keras.layers.Conv1D(32, kernel_size=3, padding='same', activation='relu')(input_layer)
    x = tf.keras.layers.AveragePooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)

    # BLSTM
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(0.3)(x)

    # Attention
    x = attention_layer(x)

    # Output
    x = GlobalAveragePooling1D()(x)
    output = Dense(output_dim, activation='linear')(x)

    model = Model(inputs=input_layer, outputs=output)

    model.compile(
        loss='mean_squared_error',
        optimizer=tf.keras.optimizers.Adam(learning_rate=alpha),
        metrics=['mae', ml_evals_util.r2_keras, ml_evals_util.rmse_keras]
    )

    model.summary()

    # === Training ===
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    ]

    model.fit(
        train_PHM_x, train_PHM_y,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1
    )

    # === Evaluation ===
    scores = model.evaluate(test_PHM_x, test_PHM_y, batch_size=batch_size, verbose=1)
    print(f" {data_identifier} - BLSTM-CNN Results:")
    print(f"  Loss  : {scores[0]:.4f}")
    print(f"  MAE   : {scores[1]:.4f}")
    print(f"  R²    : {scores[2]:.4f}")
    print(f"  RMSE  : {scores[3]:.4f}")

    y_pred = model.predict(test_PHM_x).flatten()
    print(" Prediction complete.")
    return y_pred