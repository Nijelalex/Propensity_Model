"""Data Engineering
Preprocessing the Data for creating model input
"""

from typing import Dict
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


from thesis_propensity_model.pipelines.data_engineering.preprocessing.nodes import outlier_removal, data_imputation, feature_selection_dedupe
from thesis_propensity_model.pipelines.data_engineering.data_encoding.nodes import binary_encoding, label_encoding, onehotencoding


def predict_inference(input_df: pd.DataFrame, config: Dict,config_ds: Dict, model_fit: Dict) -> pd.DataFrame:
    """Processes the data for predictions.
    Args:
        df: Inference table.
    Returns:
        train and test data
    """
    outlier_data = outlier_removal(input_df, config)
    imputed_data=data_imputation(outlier_data, config)
    feature_data=feature_selection_dedupe(imputed_data, config)
    binencoded_data=binary_encoding(feature_data, config)
    labencoded_data=label_encoding(binencoded_data, config)
    model_input=onehotencoding(labencoded_data, config)

    model_input = model_input.drop(config_ds["tgt_variable"][0], axis=1)
    customer_num=model_input[["Customer"]]
    model_input = model_input.drop("Customer", axis=1)

    standard_scaler_func = StandardScaler()
    final_table = standard_scaler_func.fit_transform(model_input)
    final_table = pd.DataFrame(final_table,columns=model_input.columns)
    model_best=model_fit.best_estimator_
    inf_predict=model_best.predict(final_table)
    inf_score=model_best.predict_proba(final_table)[:,1]
    inf_predict = pd.DataFrame({'Prediction': inf_predict,'Score': inf_score}, columns=['Prediction', 'Score'])

    logger = logging.getLogger(__name__)
    logger.info(f"The final table shape: {final_table.shape}")

    return final_table, inf_predict, customer_num


def final_inference(input_df: pd.DataFrame, score_df: pd.DataFrame, cif_df: pd.DataFrame, inf_shap: pd.DataFrame) -> pd.DataFrame:
    """Creates table to be ingested to CRM.
    
    Returns:
        CRM Table
    """
    df_spine = pd.concat([cif_df,score_df], axis=1)
    CRM_table=df_spine.merge(input_df, on="Customer", how="inner" )
    CRM_table=CRM_table.sort_values(by='Score', ascending=False)
    CRM_table["Status"]=""
    CRM_table["Description"]=""

    inf_shap['Customer']=cif_df['Customer']
    CRM_Shap=inf_shap.reindex(CRM_table.index)
    cif_df=cif_df.reindex(CRM_table.index)

    return CRM_table, CRM_Shap
