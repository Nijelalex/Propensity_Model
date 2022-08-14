"""Data Engineering
Preprocessing the Data for creating model input
"""

from typing import Dict
import logging
import pandas as pd

from sklearn.preprocessing import LabelEncoder


def binary_encoding(fs_table: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Binary Encoding.
    Args:
        df: Raw data.
    Returns:
        Preprocessed data
    """
    dic = {"yes": 1, "no": 0}
    for i in config["Encoding"]["BinaryEncoder"]:
        fs_table[i] = fs_table[i].map(dic)

    logger = logging.getLogger(__name__)
    logger.info(f"Column names are: {fs_table.columns.tolist()}")

    return fs_table


def label_encoding(df_encoded: pd.DataFrame, config: Dict):
    """Label Encoding the data
    Input: Dataframe
    Returns: Dataframe after imputation
    """
    le = LabelEncoder()
    for i in config["Encoding"]["Ordinal_Encoder"]:
        df_encoded[i] = le.fit_transform(df_encoded[i].values)

    logger = logging.getLogger(__name__)
    logger.info(f"Column names are: {df_encoded.columns.tolist()}")

    return df_encoded


def onehotencoding(df_encoded: pd.DataFrame, config: Dict):
    """Factor Analysis on Economic indicators and remove columns"""
    df_encoded = pd.get_dummies(df_encoded, columns=config["Encoding"]["Onehot_Encoder"])

    logger = logging.getLogger(__name__)
    logger.info(f"Column names are: {df_encoded.columns.tolist()}")
    logger.info(f"df shape: {df_encoded.shape}")

    return df_encoded
