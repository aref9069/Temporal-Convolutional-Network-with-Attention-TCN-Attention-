'''
- This module provides essential preprocessing functions for transforming
the raw C-MAPSS dataset into input formats suitable for sequence-based
machine learning and deep learning models.

FUNCTIONALITY:
--------------
- Normalizes sensor and operational setting features by operating condition
- Computes Remaining Useful Life (RUL) with optional capping
- Engineers cumulative threshold-based features (for multi-condition datasets)
- Generates training and test sequences for sequence-to-one models
- Aligns RUL labels with each input sequence
- Handles test label generation using truth files

NOTES:
------
- Assumes that input DataFrame includes 'id' and 'cycle' columns
- Designed for NASA C-MAPSS datasets and sequence-to-one architectures
- `common.zscore_normalize` is used for column-wise normalization



Aref Aasi, March 2024

'''


import numpy as np
import pandas as pd
import common

def normalize_features(df, group_col='setting3', feature_cols=None):
    """
    Normalize features grouped by an operating condition.
    """
    df = df.copy()
    if feature_cols is None:
        feature_cols = df.columns.drop(['id', 'cycle'])
    grouped = df.groupby(group_col)
    for col in feature_cols:
        df[col] = grouped[col].transform(common.zscore_normalize)
    return df


def compute_rul(df, max_rul=130):
    """
    Compute capped Remaining Useful Life (RUL) for each cycle.
    """
    df = df.copy()
    max_cycle = df.groupby('id')['cycle'].transform('max')
    df['RUL'] = (max_cycle - df['cycle']).clip(upper=max_rul)
    return df


def engineer_cumulative_features(df, setting_col='setting3', thresholds=None):
    """
    Add cumulative count features for specific thresholds of a given column.
    """
    if thresholds is None:
        thresholds = [0, 20, 40, 60, 80, 100]

    feature_df = pd.DataFrame(index=df.index)
    for engine_id, group in df.groupby('id'):
        cum_df = pd.DataFrame(index=group.index)
        for val in thresholds:
            cum_df[f'cumsum_{val}'] = (group[setting_col] == val).astype(int).cumsum()
        feature_df = pd.concat([feature_df, cum_df])

    return feature_df.sort_index()


def generate_sequences(df, seq_length, feature_cols):
    """
    Generate overlapping input sequences for training.
    """
    values = df[feature_cols].values
    return np.array([
        values[i:i + seq_length]
        for i in range(len(values) - seq_length + 1)
    ], dtype=np.float32)


def generate_labels(df, seq_length, label_col='RUL'):
    """
    Align labels to each input sequence.
    """
    labels = df[label_col].values
    return np.array([
        labels[i + seq_length - 1]
        for i in range(len(labels) - seq_length + 1)
    ], dtype=np.float32)


def generate_test_sequences(df, seq_length, feature_cols):
    """
    Generate last sequence for each engine in test set.
    """
    seq_list = []
    for _, group in df.groupby('id'):
        data = group[feature_cols].values
        if len(data) >= seq_length:
            seq_list.append(data[-seq_length:])
    return np.array(seq_list, dtype=np.float32)


def generate_test_labels(truth_df, test_df, max_RUL):
    """
    Create ground truth labels for test set engines.
    """
    if truth_df.shape[1] != 1:
        print(f"[Warning] Expected 1 column in truth_df, but got {truth_df.shape[1]}. Keeping only the first column.")
        truth_df = truth_df.iloc[:, [0]]

    rul_array = truth_df.values.flatten()
    test_rul = [min(r, max_RUL) for r in rul_array]
    return np.array(test_rul, dtype=np.float32)


def gen_sequence_label_pair(df, seq_length, feature_cols, label_col='RUL'):
    """
    Generate (X, y) training pairs from sequence and label columns.
    """
    x = generate_sequences(df, seq_length, feature_cols)
    y = generate_labels(df, seq_length, label_col)
    return x, y
