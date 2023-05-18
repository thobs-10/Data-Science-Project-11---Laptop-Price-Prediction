
""" 1. data from the UI of the user by making use of streamlit
2. transform the inputs
3. send them to the predictor 
4. get the prediction out put
5. return the output to the user"""

import numpy as np
import pandas as pd

from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW, SKLEARN
from zenml.pipelines import pipeline

docker_settings = DockerSettings(required_integrations=[MLFLOW, SKLEARN])
@ pipeline(enable_cache=True, settings= docker_settings)
def inference_pipeline(
    get_input,
    preprocess_input,
    prediction_service_loader,
    predictor
):
    data = get_input()
    transformed_data = preprocess_input(data)
    model_deployment_service = prediction_service_loader()
    prediction = predictor(model_deployment_service,transformed_data)
