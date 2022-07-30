"""Example code for the nodes in the example pipeline. This code is meant
just for illustrating basic Kedro features.
Delete this when you start working on your own Kedro project.
"""

from kedro.pipeline import node, pipeline
from .nodes import preprocess


def create_pipeline(**kwargs):
    return pipeline(
        [
            node(
                func=preprocess,
                inputs=["bank_raw","params:Outlier_params"],
                outputs="model_input_table",
                name="preprocessing",
            )
        ],
        
    )