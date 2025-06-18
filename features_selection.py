'''

- This module provides a full preprocessing pipeline for the NASA C-MAPSS 
dataset, tailored for sequence-to-one regression models (e.g., CNN, LSTM, 
Transformer). It prepares training and test data by computing RUL labels, 
normalizing features, engineering optional cumulative features, and 
generating fixed-length input sequences.

FEATURES:
---------
- Computes capped Remaining Useful Life (RUL) labels for training and test data
- Normalizes features by operating condition (`setting3`) for better generalization
- Supports both single and multi-operating condition datasets (FD001-FD004)
- Automatically engineers cumulative features for multi-condition settings
- Generates overlapping input sequences and aligned labels for supervised learning
- Pads training data to fit the model's batch size for consistent training
- Output is transposed to (features, sequence, samples) format for deep learning models

ASSUMPTIONS:
------------
- The user is familiar with sequence-to-one learning (e.g., RNNs or CNNs on time-series)
- Input data uses standard C-MAPSS format, already loaded into DataFrames
- RUL is capped to avoid label skewing from outliers in long time-to-failure cases

USAGE:
------
- Used as part of the preprocessing phase in an RUL prediction pipeline
- Designed to be called with `train_df`, `test_df`, `truth_df`, and `data_ml_settings`


-----
Aref Aasi, May 2024

'''

import numpy as np
import pandas as pd
import sys
sys.path.append("..") # Adds higher directory to python modules path.

import common, preprocessing

###########################################
# A. DEFINE FUNCTIONS FOR DATA PREPROCESSING
###########################################



def preprocess_data(train_df, test_df, truth_df, data_ml_settings):
    """
    Preprocess C-MAPSS data for sequence-to-one RUL prediction models.

    Args:
        train_df (pd.DataFrame): Raw training set.
        test_df (pd.DataFrame): Raw test set.
        truth_df (pd.DataFrame): True RUL values for test engines.
        data_ml_settings (dict): Dictionary with 'max_RUL', 'sequence_length', and 'batch_size'.

    Returns:
        train_x (np.ndarray): Training data (shape: features, sequence, samples).
        train_y (np.ndarray): RUL labels for training data.
        test_x (np.ndarray): Test data.
        test_y (np.ndarray): RUL labels for test data.
    """
    max_RUL = data_ml_settings['max_RUL']
    seq_len = data_ml_settings['sequence_length']
    batch_size = data_ml_settings['batch_size']

    # Step 1: Clean and adjust truth_df
    truth_df = truth_df.dropna(axis=1, how='all').copy()
    if truth_df.shape[1] != 1:
        print(f"[Warning] Expected 1 column in truth_df, but got {truth_df.shape[1]}. Keeping only the first column.")
        truth_df = truth_df.iloc[:, [0]]
    truth_df.columns = ['RUL']
    truth_df['id'] = truth_df.index + 1

    # Step 2: Label training and test RUL
    train_df['RUL'] = train_df.groupby('id')['cycle'].transform(lambda x: (x.max() - x).clip(upper=max_RUL))

    last_cycles = test_df.groupby('id')['cycle'].max().reset_index().rename(columns={'cycle': 'last'})
    truth_df = pd.merge(truth_df, last_cycles, on='id')
    truth_df['max'] = truth_df['RUL'] + truth_df['last']
    test_df = pd.merge(test_df, truth_df[['id', 'max']], on='id', how='left')
    test_df['RUL'] = (test_df['max'] - test_df['cycle']).clip(upper=max_RUL)
    test_df.drop(columns=['max'], inplace=True)

    # Step 3: Normalize based on operating condition (setting3)
    combined = pd.concat([train_df, test_df])
    feature_cols = [col for col in combined.columns if col not in ['id', 'cycle', 'RUL']]
    combined[feature_cols] = combined.groupby('setting3')[feature_cols].transform(common.find_col_norm)

    train_norm = combined.iloc[:len(train_df)].copy()
    test_norm = combined.iloc[len(train_df):].copy()
    train_norm['RUL'] = train_df['RUL'] / max_RUL
    test_norm['RUL'] = test_df['RUL'] / max_RUL

    # Step 4: Add cumulative features for multi-condition datasets
    if train_df['setting3'].nunique() > 1:
        print("--multi operating conditions--")
        train_cum = preprocessing.engineer_cumulative_features(train_df).apply(common.find_col_norm)
        test_cum = preprocessing.engineer_cumulative_features(test_df).apply(common.find_col_norm)
        train_norm = pd.concat([train_norm, train_cum], axis=1)
        test_norm = pd.concat([test_norm, test_cum], axis=1)
    else:
        print("--single operating condition--")

    # Step 5: Sequence generation
    final_features = [col for col in train_norm.columns if col not in ['id', 'cycle', 'RUL']]
    train_x = preprocessing.generate_sequences(train_norm, seq_len, final_features)
    train_y = preprocessing.generate_labels(train_norm, seq_len)

    test_x = preprocessing.generate_test_sequences(test_norm, seq_len, final_features)
    test_y = preprocessing.generate_test_labels(truth_df[['id', 'RUL']], test_df, max_RUL)

    # Step 6: Pad to batch_size
    pad_len = batch_size - (train_x.shape[0] % batch_size)
    if pad_len != batch_size:
        train_x = np.concatenate([train_x, train_x[:pad_len]], axis=0)
        train_y = np.concatenate([train_y, train_y[:pad_len]], axis=0)

    # Final shape adjustment
    return train_x.transpose(2, 1, 0), train_y, test_x.transpose(2, 1, 0), test_y
