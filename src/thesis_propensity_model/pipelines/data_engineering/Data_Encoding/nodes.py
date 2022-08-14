"""Data Engineering
Preprocessing the Data for creating model input
"""

from typing import Any, Dict

import pandas as pd
import logging
from sklearn.preprocessing import LabelEncoder


def binary_encoding(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Binary Encoding.
    Args:
        df: Raw data.
    Returns:
        Preprocessed data
    """
    dic = {"yes": 1, "no": 0}
    for i in config["Encoding"]["BinaryEncoder"]:
        df[i] = df[i].map(dic)

    logger = logging.getLogger(__name__)
    logger.info(f"Column names are: {df.columns.tolist()}")

    return df


def label_encoding(df: pd.DataFrame, config: Dict):
    """Label Encoding the data
    Input: Dataframe
    Returns: Dataframe after imputation
    """
    le = LabelEncoder()
    for i in config["Encoding"]["Ordinal_Encoder"]:
        df[i] = le.fit_transform(df[i].values)

    logger = logging.getLogger(__name__)
    logger.info(f"Column names are: {df.columns.tolist()}")

    return df


def onehotencoding(df: pd.DataFrame, config: Dict):
    """Factor Analysis on Economic indicators and remove columns"""
    df = pd.get_dummies(df, columns=config["Encoding"]["Onehot_Encoder"])

    logger = logging.getLogger(__name__)
    logger.info(f"Column names are: {df.columns.tolist()}")
    logger.info(f"df shape: {df.shape}")

    return df
