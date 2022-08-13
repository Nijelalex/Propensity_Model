"""Data Preprocessing Node
"""

from kedro.pipeline import node, pipeline
from .nodes import outlier_removal,data_imputation,feature_selection_dedupe


def create_pipeline(**kwargs):
    return pipeline(
        [
            node(
                func=outlier_removal,
                inputs=["bank_raw","params:de_params"],
                outputs="outlier_removed_table",
                name="Outlier_Removal",
                tags="de",
            ),
            node(
                func=data_imputation,
                inputs=["outlier_removed_table","params:de_params"],
                outputs="imputed_table",
                name="Data_Imputation",
                tags="de",
            ),
            node(
                func=feature_selection_dedupe,
                inputs=["imputed_table","params:de_params"],
                outputs="fs_table",
                name="Feature_Selection",
                tags="de",
            ),
        ],
        
    )