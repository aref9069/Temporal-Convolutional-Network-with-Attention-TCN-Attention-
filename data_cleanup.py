'''
- This module provides utilities to load, validate, and clean NASA's
C-MAPSS datasets for Remaining Useful Life (RUL) prediction tasks.
It reads training data (run-to-failure), test data (non-failed units),
and corresponding RUL labels, standardizes their formats, and outputs
structured DataFrames for further preprocessing and modeling.

FEATURES:
---------
- Supports FD001-FD004 sub-datasets via the `data_identifier` flag
- Cleans raw files by selecting valid columns and assigning names
- Sorts data by engine ID and cycle for consistent time ordering
- Handles inconsistencies in truth files with multiple columns
- Returns cleaned pandas DataFrames for train, test, and truth data

ASSUMPTIONS:
------------
- The dataset folder contains files in standard C-MAPSS naming format:
    train_FD00X.txt, test_FD00X.txt, RUL_FD00X.txt
- Each file follows a whitespace-delimited structure
- All sensors and settings follow the predefined 26-column format

Aref Aasi, April 2024


'''

import pandas as pd
import os

def ingest_and_cleanup_data(data_dir, data_identifier):
    """
    Load and clean C-MAPSS train/test datasets and ground truth RUL.

    Args:
        data_dir (str): Directory path to the dataset files
        data_identifier (str): Dataset identifier (e.g. 'FD001', 'FD002', etc.)

    Returns:
        tuple:
            train_df (pd.DataFrame): Run-to-failure sensor data
            test_df (pd.DataFrame): Operational sensor data without failure
            truth_df (pd.DataFrame): Ground truth RUL values for test data
    """

    # -------------------------------
    # File Paths
    # -------------------------------
    train_path = os.path.join(data_dir, f'train_{data_identifier}.txt')
    test_path = os.path.join(data_dir, f'test_{data_identifier}.txt')
    rul_path = os.path.join(data_dir, f'RUL_{data_identifier}.txt')

    # -------------------------------
    # Common column names
    # -------------------------------
    columns = [
        'id', 'cycle', 'setting1', 'setting2', 'setting3',
        's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10',
        's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21'
    ]

    # -------------------------------
    # Load and clean training data
    # -------------------------------
    train_df = pd.read_csv(train_path, sep=r'\s+', header=None)
    train_df = train_df.iloc[:, :len(columns)]
    train_df.columns = columns
    train_df = train_df.sort_values(['id', 'cycle'])

    # -------------------------------
    # Load and clean test data
    # -------------------------------
    test_df = pd.read_csv(test_path, sep=r'\s+', header=None)
    test_df = test_df.iloc[:, :len(columns)]
    test_df.columns = columns
    test_df = test_df.sort_values(['id', 'cycle'])

    # -------------------------------
    # Load and validate ground truth RUL
    # -------------------------------
    truth_df = pd.read_csv(rul_path, sep=r'\s+', header=None)

    # Forcefully reduce to first column only, regardless of how many exist
    if truth_df.shape[1] > 1:
        print(f"[Warning] Expected 1 column in truth_df, but got {truth_df.shape[1]}. Using only the first column.")
        truth_df = truth_df.iloc[:, [0]]  # Keep only the first column

    # Explicitly rename the column to ensure clarity and consistency
    truth_df.columns = ['RUL']

    # -------------------------------
    # Info
    # -------------------------------
    print(f" Loaded: {data_identifier}")
    print(f" train_df: {train_df.shape}")
    print(f" test_df:  {test_df.shape}")
    print(f" truth_df: {truth_df.shape}")

    return train_df, test_df, truth_df
