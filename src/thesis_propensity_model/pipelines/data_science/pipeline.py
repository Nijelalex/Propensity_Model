"""Data Engineering Pipeline
"""


from thesis_propensity_model.pipelines.data_science.Model_input.pipeline import (
    model_input_pipeline,
)
from kedro.pipeline import pipeline, Pipeline


def create_ds_pipeline():

    return pipeline(
        pipe=Pipeline(
            [
                model_input_pipeline(),
            ],
            tags="ds_pipeline",
        )
    )
