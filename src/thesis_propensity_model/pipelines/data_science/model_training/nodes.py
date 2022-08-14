"""Model Training
Training the model
"""

from typing import Dict
import logging
import pandas as pd



def split_data(input_df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Splits the data into train-test.
    Args:
        df: Model input.
    Returns:
        train and test data
    """
    x_data = input_df.drop(config["tgt_variable"][0], axis=1)
    y_data = input_df[config["tgt_variable"][0]]
    stratified_shuffle = StratifiedShuffleSplit(
        n_splits=1, test_size=0.3, random_state=1
    )
    for train_index, test_index in stratified_shuffle.split(x_data, y_data):
        train_df = input_df.iloc[train_index]
        test_df = input_df.iloc[test_index]

    logger = logging.getLogger(__name__)
    logger.info(f"Column names are: {train_df.shape,test_df.shape}")

    return train_df, test_df


def standard_scaler(
    train_df: pd.DataFrame, test_df: pd.DataFrame, config: Dict
) -> pd.DataFrame:
    """Splits the data into train-test.
    Args:
        df: Train_Df.
        df1: Test_Df.
        config: ds parameters
    Returns:
        train and test data
    """

    # Get Feature table and target table
    x_train = train_df.drop(config["tgt_variable"][0], axis=1)
    y_train = train_df[config["tgt_variable"][0]]
    x_test = test_df.drop(config["tgt_variable"][0], axis=1)
    y_test = test_df[config["tgt_variable"][0]]

    standard_scaler_func = StandardScaler()
    x_train_s = standard_scaler_func.fit_transform(x_train)
    x_test_s = standard_scaler_func.transform(x_test)

    logger = logging.getLogger(__name__)
    logger.info(
        f"Column names are: {x_train.shape,y_train.shape, x_train_s.shape, x_test_s.shape}"
    )

    return x_train_s, y_train, x_test_s, y_test


def class_imbalance(x_train_s: pd.DataFrame, y_train: pd.DataFrame) -> pd.DataFrame:
    """Splits the data into train-test.
    Args:
        df: X train scaled.
        df1: Y train
    Returns:
        X_smote, Y_smote
    """
    smote = SMOTE()
    x_smote, y_smote = smote.fit_resample(x_train_s, y_train)

    logger = logging.getLogger(__name__)
    logger.info(f"Column names are: {x_smote.shape,y_smote.shape}")

    return x_smote, y_smote
