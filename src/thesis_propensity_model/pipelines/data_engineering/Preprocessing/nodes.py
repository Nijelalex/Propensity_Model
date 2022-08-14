"""Data Engineering
Preprocessing the Data for creating model input
"""

from typing import Any, Dict

import pandas as pd
import logging
import numpy as np
from sklearn.decomposition import FactorAnalysis


def imputation(x: pd.Series) -> pd.Series:
    # Impute by the mode
    x = x.replace("unknown", np.nan)
    x = x.fillna(x.mode()[0])
    return x


def outlier_removal(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Removes the outliers from data.
    Args:
        df: Raw data.
    Returns:
        Preprocessed data
    """
    bank_int = df.copy()

    # Remove outliers for some features as defined in the parameters
    for k, v in config["Outlier_params"].items():
        bank_int = bank_int[~(bank_int[k] > v)]

    logger = logging.getLogger(__name__)
    logger.info(f"Column names are: {bank_int.columns.tolist()}")

    return bank_int


def data_imputation(df: pd.DataFrame, config: Dict):
    """Impute the nulls in the data
    Input: Dataframe
    Returns: Dataframe after imputation
    """
    for col in config["Imputation_columns"]:
        df[col] = imputation(df[col])

    logger = logging.getLogger(__name__)
    logger.info(f"Column names are: {df.columns.tolist()}")

    return df


def feature_selection_dedupe(df: pd.DataFrame, config: Dict):
    """Factor Analysis on Economic indicators and remove columns"""
    df["Economic_Indicators"] = FactorAnalysis(n_components=1).fit_transform(
        df[config["Feature_Selection"]]
    )
    df = df.drop(config["Feature_Selection"], axis=1)
    df = df.drop_duplicates()
    df.rename(columns={"y": "deposit"}, inplace=True)

    logger = logging.getLogger(__name__)
    logger.info(f"Column names are: {df.columns.tolist()}")
    logger.info(f"df shape: {df.shape}")

    return df
