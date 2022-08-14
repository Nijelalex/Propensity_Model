"""Data Engineering
Preprocessing the Data for creating model input
"""

from typing import Dict
import logging
import pandas as pd
import numpy as np
from sklearn.decomposition import FactorAnalysis


def imputation(x_col: pd.Series) -> pd.Series:
    """Imputation Node"""
    # Impute by the mode
    x_col = x_col.replace("unknown", np.nan)
    x_col = x_col.fillna(x_col.mode()[0])
    return x_col


def outlier_removal(raw_data: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Removes the outliers from data.
    Args:
        df: Raw data.
    Returns:
        Preprocessed data
    """
    bank_int = raw_data.copy()

    # Remove outliers for some features as defined in the parameters
    for key_val, val in config["Outlier_params"].items():
        bank_int = bank_int[~(bank_int[key_val] > val)]

    logger = logging.getLogger(__name__)
    logger.info(f"Column names are: {bank_int.columns.tolist()}")

    return bank_int


def data_imputation(outlier_data: pd.DataFrame, config: Dict):
    """Impute the nulls in the data
    Input: Dataframe
    Returns: Dataframe after imputation
    """
    for col in config["Imputation_columns"]:
        outlier_data[col] = imputation(outlier_data[col])

    logger = logging.getLogger(__name__)
    logger.info(f"Column names are: {outlier_data.columns.tolist()}")

    return outlier_data


def feature_selection_dedupe(impute_data: pd.DataFrame, config: Dict):
    """Factor Analysis on Economic indicators and remove columns"""
    impute_data["Economic_Indicators"] = FactorAnalysis(n_components=1).fit_transform(
        impute_data[config["Feature_Selection"]]
    )
    impute_data = impute_data.drop(config["Feature_Selection"], axis=1)
    impute_data = impute_data.drop_duplicates()
    impute_data.rename(columns={"y": "deposit"}, inplace=True)

    logger = logging.getLogger(__name__)
    logger.info(f"Column names are: {impute_data.columns.tolist()}")
    logger.info(f"df shape: {impute_data.shape}")

    return impute_data
