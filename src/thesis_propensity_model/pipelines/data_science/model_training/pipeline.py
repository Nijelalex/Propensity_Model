"""Model Input Pipeline
"""

from kedro.pipeline import node, pipeline
from .nodes import fit_xgb, fit_logistic_regression, fit_knn, fit_svc, fit_nb


def model_training_pipeline():
    """Model input pipeline"""
    return pipeline(
        [
            node(
                func=fit_logistic_regression,
                inputs=["X_smote", "Y_smote", "params:ds_params"],
                outputs="lr_model",
                name="lr_model",
                tags="ds",
            ),
            node(
                func=fit_knn,
                inputs=["X_smote", "Y_smote", "params:ds_params"],
                outputs="knn_model",
                name="knn_model",
                tags="ds",
            ),
            node(
                func=fit_svc,
                inputs=["X_smote", "Y_smote", "params:ds_params"],
                outputs="svc_model",
                name="svc_model",
                tags="ds",
            ),
            node(
                func=fit_nb,
                inputs=["X_smote", "Y_smote", "params:ds_params"],
                outputs="nb_model",
                name="nb_model",
                tags="ds",
            ),
            node(
                func=fit_xgb,
                inputs=["X_smote", "Y_smote", "params:ds_params"],
                outputs="xgb_model",
                name="xgb_model",
                tags="ds",
            ),
            
        ],
    )
