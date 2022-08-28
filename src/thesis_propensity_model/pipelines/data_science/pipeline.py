"""Data Engineering Pipeline
"""

from kedro.pipeline import pipeline, Pipeline
from thesis_propensity_model.pipelines.data_science.model_input.pipeline import (
    model_input_pipeline,
)
from thesis_propensity_model.pipelines.data_science.model_training.pipeline import (
    model_training_pipeline,
)

from thesis_propensity_model.pipelines.data_science.inference.pipeline import (
    inference_pipeline,
)

def create_ds_pipeline():
    """Create DS pipeline"""
    return pipeline(
        pipe=Pipeline(
            [
                model_input_pipeline(),
                model_training_pipeline(),
                inference_pipeline(),

            ],
            tags="ds_pipeline",
        )
    )
