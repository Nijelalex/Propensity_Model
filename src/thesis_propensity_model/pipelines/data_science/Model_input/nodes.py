"""Data Engineering
Preprocessing the Data for creating model input
"""

from typing import Any, Dict

import pandas as pd
import logging
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
import imblearn
from imblearn.over_sampling import SMOTE



def split_data(df: pd.DataFrame,config: Dict) -> pd.DataFrame:
    """Splits the data into train-test.
    Args:
        df: Model input.
    Returns:
        train and test data
    """
    X=df.drop(config['tgt_variable'][0],axis=1)
    Y=df[config['tgt_variable'][0]]
    sss = StratifiedShuffleSplit(n_splits=1,test_size=0.3,random_state=1)
    for train_index,test_index in sss.split(X,Y):
        train_df = df.iloc[train_index]
        test_df = df.iloc[test_index]    
    
    logger = logging.getLogger(__name__)
    logger.info(f"Column names are: {train_df.shape,test_df.shape}")  
    
    return train_df,test_df

def standard_scaler(df: pd.DataFrame,df1: pd.DataFrame,config: Dict) -> pd.DataFrame:
    """Splits the data into train-test.
    Args:
        df: Train_Df.
        df1: Test_Df.
        config: ds parameters
    Returns:
        train and test data
    """

    #Get Feature table and target table
    X_train = df.drop(config['tgt_variable'][0],axis=1)
    Y_train = df[config['tgt_variable'][0]]
    X_test = df1.drop(config['tgt_variable'][0],axis=1)
    Y_test = df1[config['tgt_variable'][0]] 
    
    ss = StandardScaler()
    X_train_s = ss.fit_transform(X_train)
    X_test_s = ss.transform(X_test)   
    
    logger = logging.getLogger(__name__)
    logger.info(f"Column names are: {X_train.shape,Y_train.shape,X_test.shape,Y_test.shape, X_train_s.shape, X_test_s.shape}")  
    
    return X_train_s,Y_train,X_test_s,Y_test


def class_imbalance(df: pd.DataFrame,df1: pd.DataFrame,config: Dict) -> pd.DataFrame:
    """Splits the data into train-test.
    Args:
        df: X train scaled.
        df1: Y train
    Returns:
        X_smote, Y_smote
    """
    smote=SMOTE()
    X_smote,Y_smote=smote.fit_resample(df, df1)

    logger = logging.getLogger(__name__)
    logger.info(f"Column names are: {X_smote.shape,Y_smote.shape}")  
    
    return X_smote,Y_smote