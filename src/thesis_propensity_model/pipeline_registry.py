"""Project pipelines."""
import imp
from typing import Dict

from kedro.pipeline import Pipeline
from thesis_propensity_model.pipelines import data_engineering as de
from thesis_propensity_model.pipelines import data_science as ds


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:s
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    data_engineering_pipeline = de.create_de_pipeline()
    data_science_pipeline = ds.create_ds_pipeline()

    return {
        "de": data_engineering_pipeline,
        "ds": data_science_pipeline,
        "overall": data_engineering_pipeline + data_science_pipeline,
        "__default__": data_engineering_pipeline + data_science_pipeline,
    }
