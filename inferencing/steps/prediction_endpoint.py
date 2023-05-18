import numpy as np
import pandas as pd
from zenml.steps import step
import requests
import json
from typing import Tuple

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from zenml.integrations.bentoml.services.bentoml_deployment import BentoMLDeploymentEndpoint
from zenml.integrations.bentoml.services import BentoMLDeploymentService

SERVICE_URL = "http://localhost:3000/predict"
@step
def preprocess_input_data(input_data:pd.DataFrame,X_train:pd.DataFrame) -> np.ndarray:
    '''preprocess the input from a dataframe to an array'''
    # query = np.array(input_data)
    # query.reshape(1,12)
    column_transformer = ColumnTransformer(transformers=[
            ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
        ],remainder='passthrough')
    X_train_tranformed = np.array(column_transformer.fit_transform(X_train))
    input_data_tranformed = np.array(column_transformer.transform(input_data))

    return input_data_tranformed

@step
def make_inference(
    input_data:np.ndarray,
    # service_url:str,
    ) -> str:
    '''make inference from the api'''
   
    serialized_input_data = json.dumps(input_data.tolist())
    response = requests.post(
        SERVICE_URL,
        data=serialized_input_data,
        headers={"content-type":"application/json"}
    )
    return response.text

@step
def write_predictions(predictions:str, expected_outputs:pd.DataFrame)-> None:
    print("Predictions vs Expected Output")
    print(f"{predictions}   {expected_outputs.values}")



