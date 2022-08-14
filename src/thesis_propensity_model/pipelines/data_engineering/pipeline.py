"""Data Engineering Pipeline
"""


from thesis_propensity_model.pipelines.data_engineering.Preprocessing.pipeline import (
    create_preprocess_pipeline,
)
from thesis_propensity_model.pipelines.data_engineering.Data_Encoding.pipeline import (
    create_encoding_pipeline,
)
from kedro.pipeline import pipeline, Pipeline


def create_de_pipeline():

    return pipeline(
        pipe=Pipeline(
            [
                create_preprocess_pipeline(),
                create_encoding_pipeline(),
            ],
            tags="de_pipeline",
        )
    )
