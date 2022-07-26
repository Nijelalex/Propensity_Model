"""Model Training and evaluating Pipeline
"""

from kedro.pipeline import node, pipeline
from .nodes import fit_model, final_model, evaluate_model, explainability


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
            node(
                func=fit_model,
                inputs=["X_smote", "Y_smote", "params:ds_params1"],
                outputs="fit_model1",
                name="fit_model2",
                tags="ds",
            ),
            node(
                func=final_model,
                inputs=["X_smote", "Y_smote", "params:ds_params1","fit_model", "fit_model1"],
                outputs="final_model",
                name="final_model",
                tags="ds",
            ),
            node(
                func=evaluate_model,
                inputs=["final_model","X_smote","Y_smote", "X_test_scaled","Y_test","params:ds_params"],
                outputs=["performance_metric","confusion_matrix","classification_report"],
                name="evaluate_model",
                tags="ds",
            ),
            node(
                func=explainability,
                inputs=["fit_model","X_test_scaled","train_df","params:ds_params"],
                outputs="shap_val",
                name="model_explain",
                tags="ds",
            ),
        ],
    )
