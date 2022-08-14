"""Data Engineering Pipeline
"""

from kedro.pipeline import pipeline, Pipeline
from thesis_propensity_model.pipelines.data_science.model_input.pipeline import (
    model_input_pipeline,
)


def create_ds_pipeline():
    """Create DS pipeline"""
    return pipeline(
        pipe=Pipeline(
            [
                model_input_pipeline(),
            ],
            tags="ds_pipeline",
        )
    )
