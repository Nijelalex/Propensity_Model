"""Model Input Pipeline
"""

from kedro.pipeline import node, pipeline
from .nodes import split_data, standard_scaler, class_imbalance


def model_input_pipeline(**kwargs):
    return pipeline(
        [
            
            node(
                func=split_data,
                inputs=["model_input_table","params:ds_params"],
                outputs=["train_df","test_df"],
                name="Train-Test_split",
                tags="ds",
            ),
            node(
                func=standard_scaler,
                inputs=["train_df","test_df","params:ds_params"],
                outputs=["X_train_scaled","Y_train","X_test_scaled","Y_test"],
                name="Scaled_data",
                tags="ds",
            ),
            node(
                func=class_imbalance,
                inputs=["X_train_scaled","Y_train","params:ds_params"],
                outputs=["X_smote","Y_smote"],
                name="class_imbalance",
                tags="ds",
            ),
        ],
        
    )