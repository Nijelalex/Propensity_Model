"""Data Engineering Pipeline
"""

from kedro.pipeline import pipeline, Pipeline
from thesis_propensity_model.pipelines.data_engineering.preprocessing.pipeline import (
    create_preprocess_pipeline,
)
from thesis_propensity_model.pipelines.data_engineering.data_encoding.pipeline import (
    create_encoding_pipeline,
)



def create_de_pipeline():
    """The De pipeline that runs preprocess and encoding pipelines"""
    return pipeline(
        pipe=Pipeline(
            [
                create_preprocess_pipeline(),
                create_encoding_pipeline(),
            ],
            tags="de_pipeline",
        )
    )
