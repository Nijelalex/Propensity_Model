"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline
from thesis_propensity_model.pipelines import data_engineering as de


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    data_engineering_pipeline = de.create_pipeline()


    return {
        "de":data_engineering_pipeline,
        "__default__": data_engineering_pipeline,
        }
