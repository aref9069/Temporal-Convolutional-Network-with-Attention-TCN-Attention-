'''
- Preprocess raw dataset 

ASSUMPTIONS: 
    - user must have a familiarity with Time-series dataset
    - this snippet was developed based on the data description

USAGE:

   NASA C-MAPPS Dataset

Aref Aasi, January 2024


'''


import numpy as np
import pandas as pd
import sys
sys.path.append("..") # Adds higher directory to python modules path.

###########################################
# A. DEFINE FUNCTIONS FOR DATA PREPROCESSING
###########################################


def ingest_and_cleanup_data(data_dir, data_identifier):
    """Ingest the Data, Do cleanup (if necessary) and converts into dataframe.

    Args:
    data_dir: Full path to directory containing the dataset
    data_identifier: Dataset identifier e.g. FD001.

    Returns:
    train_df: Training dataset dataframe
    test_df: Test dataset dataframe
    truth_df: Ground Truth dataset dataframe
    """ 


    ##################################
    # A.1 Data Ingestion
    ##################################

    # read training data - It is the aircraft engine run-to-failure data.
    train_df = pd.read_csv(data_dir + '/train_'+data_identifier+'.txt', sep=" ", header=None)
    train_df.drop(train_df.columns[[26, 27]], axis=1, inplace=True)
    train_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                         's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                         's15', 's16', 's17', 's18', 's19', 's20', 's21']

    train_df = train_df.sort_values(['id','cycle'])

    # read test data - It is the aircraft engine operating data without failure events recorded.
    test_df = pd.read_csv(data_dir + '/test_'+data_identifier+'.txt', sep=" ", header=None)
    test_df.drop(test_df.columns[[26, 27]], axis=1, inplace=True)
    test_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                         's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                         's15', 's16', 's17', 's18', 's19', 's20', 's21']

    # read ground truth data - It contains the information of true remaining cycles for each engine in the testing data.
    truth_df = pd.read_csv(data_dir + '/RUL_'+data_identifier+'.txt', sep=" ", header=None)
    truth_df.drop(truth_df.columns[[1]], axis=1, inplace=True)

    print('train_df shape : ', train_df.shape)
    print('test_df shape : ', test_df.shape)

    return train_df, test_df, truth_df
