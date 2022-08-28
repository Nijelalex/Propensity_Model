"""Inference Pipeline
"""

from kedro.pipeline import node, pipeline
from .nodes import predict_inference, final_inference


def inference_pipeline():
    """     Inference pipeline"""
    return pipeline(
        [
            node(
                func=predict_inference,
                inputs=["inference_data_db", "params:de_params","params:ds_params","fit_model"],
                outputs=["inference_final","inf_predict","customer_final"],
                name="predict_inference",
                tags="inf",
            ),
            node(
                func=final_inference,
                inputs=["inference_data_db", "inf_predict","customer_final","inference_final"],
                outputs=["crm_input","crm_shap"],
                name="CRM_input",
                tags="inf",
            ),
        ],
    )
