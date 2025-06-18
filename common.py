'''

- This module provides helper functions used for preparing and cleaning 
time-series sensor data prior to training machine learning models on 
Remaining Useful Life (RUL) tasks, particularly with the NASA C-MAPSS dataset.

-----------------------------------------------------------------

FUNCTIONS INCLUDED:
- Z-score normalization with standard or robust (clipped) methods.
- Zero-value handling in time-series using backward fill.
- Column-wise normalization function for grouped operations.
- Row indexing utility for DataFrame operations.

ASSUMPTIONS:
- Input data is in the form of pandas Series or DataFrames.
- Designed for use with engine degradation datasets in sequence-to-one prediction settings.

USAGE:
- Import this module in preprocessing scripts or pipelines where data normalization and cleaning are required.
- Particularly suited for C-MAPSS RUL datasets or similar sensor time-series tasks.


Aref Aasi, March 2024

'''



import numpy as np
import pandas as pd

def zscore_normalize(series):
    """
    Apply z-score normalization to a pandas Series.
    If the standard deviation is zero, avoids division by zero.

    Args:
        series (pd.Series): The input column.

    Returns:
        pd.Series: Normalized series.
    """
    mean = series.mean()
    std = series.std()
    if std == 0 or np.isnan(std):
        std = 1.0
    return (series - mean) / std


def robust_zscore_normalize(series, clip_range=None):
    """
    Z-score normalization with optional clipping for outlier resistance.

    Args:
        series (pd.Series): Input column.
        clip_range (tuple): Optional (min, max) range to clip values.

    Returns:
        pd.Series: Normalized and optionally clipped series.
    """
    normed = zscore_normalize(series)
    if clip_range:
        return normed.clip(lower=clip_range[0], upper=clip_range[1])
    return normed


def clean_zeros_bfill(series):
    """
    Replace internal zeros using backward fill, keep leading zeros intact,
    and extend the last non-zero value forward to the end.

    Args:
        series (pd.Series): Input time-series column.

    Returns:
        pd.Series: Cleaned series.
    """
    non_zero_indices = series.to_numpy().nonzero()[0]
    if len(non_zero_indices) == 0:
        return series

    first = non_zero_indices[0]
    last = non_zero_indices[-1]

    filled = series.copy()
    filled.iloc[first:last+1] = filled.iloc[first:last+1].replace(to_replace=0, method='bfill')
    filled.iloc[:first] = 0
    filled.iloc[last+1:] = filled.iloc[last]
    
    return filled


def get_row_index(row):
    """
    Return the index of a DataFrame row.

    Args:
        row (pd.Series): Row object.

    Returns:
        int: Row index.
    """
    return row.name


def find_col_norm(x):
    """
    Column-wise normalization. If std is zero, divide by 1 instead.
    """
    std = x.std()
    return (x - x.mean()) / (std if std != 0 else 1)
