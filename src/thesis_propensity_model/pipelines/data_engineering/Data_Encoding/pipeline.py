"""Data Preprocessing Node
"""

from kedro.pipeline import node, pipeline
from .nodes import binary_encoding,label_encoding,onehotencoding


def create_encoding_pipeline(**kwargs):
    return pipeline(
        [
            node(
                func=binary_encoding,
                inputs=["fs_table","params:de_params"],
                outputs="binary_encoded_data",
                name="Binary_encoding",
                tags="de",
            ),
            node(
                func=label_encoding,
                inputs=["binary_encoded_data","params:de_params"],
                outputs="label_encoded_data",
                name="Label_Encoding",
                tags="de",
            ),
            node(
                func=onehotencoding,
                inputs=["label_encoded_data","params:de_params"],
                outputs="model_input_table",
                name="One_Hot_Encoding",
                tags="de",
            ),
            
        ],
        
    )