"""Model Training and evaluating Pipeline
"""

from kedro.pipeline import node, pipeline
from .nodes import fit_model


def model_training_pipeline():
    """Model input pipeline"""
    return pipeline(
        [
            node(
                func=fit_model,
                inputs=["X_smote", "Y_smote", "params:ds_params"],
                outputs="fit_model",
                name="fit_model",
                tags="ds",
            ),
            
        ],
    )
