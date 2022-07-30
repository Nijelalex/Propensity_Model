"""Data Engineering
Preprocessing the Data for creating model input
"""

from typing import Any, Dict

import pandas as pd
import logging
import numpy as np

# categoricals: gender, region code, driving license, previously insured,
#   vehicle damage,
# ordinal: vehicle age

# def trim_outliers(x: pd.Series, q1=0.05, q3=0.95) -> pd.Series:
#     quartile1 = x.quantile(q1)
#     quartile3 = x.quantile(q3)
#     interquantile_range = quartile3 - quartile1
#     up_limit = quartile3 + 3 * interquantile_range
#     low_limit = quartile1 - 3 * interquantile_range

#     # cap (tukey's fences)
#     return x.clip(lower=low_limit, upper=up_limit)


# def ordinal_vehicle_age(x_veh_age: pd.Series) -> pd.Series: 
#     x_veh_age.loc[x_veh_age == "< 1 Year"] = 1
#     x_veh_age.loc[x_veh_age == "1-2 Year"] = 2
#     x_veh_age.loc[x_veh_age == "> 2 Years"] = 3
#     x_veh_age = x_veh_age.fillna(0)
#     return x_veh_age


def preprocess(df: pd.DataFrame,config: Dict) -> pd.DataFrame:
    """Preprocesses the data.
    Args:
        df: Raw data.
    Returns:
        Preprocessed data
    """
    bank_int=df.copy()
    for k,v in config.items():
        bank_int=bank_int[~(bank_int[k]>v)]
    logger = logging.getLogger(__name__)
    logger.info(f"Column names are: {bank_int.columns.tolist()}")
    return bank_int