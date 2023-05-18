import numpy as np
import pandas as pd
from zenml.steps import step, Output, step_output
from datetime import datetime

import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from zenml.client import Client
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.experiment_trackers import (
    MLFlowExperimentTracker,
)
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
# get the stack experiment tracking uri
MLFLOW_TRACKING_URI = get_tracking_uri()
# set the stack experiment tracking uri to be the current working one
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
#get the experiment tracker of the active stack from the stack components
experiment_tracker = Client().active_stack.experiment_tracker
# the backend trcking server
MLFLOW_TRACKING_SERVER = "sqlite:///laptop.db"
EXPERIMENT_NAME = "HPO experiment"

#model_name = "laptop-predicition-modelV1"
stage="Production"
model_name ="laptop-price-model"
THRESHOLD = 0.85


from zenml.steps import BaseParameters, step

class DeploymentTriggerParameters(BaseParameters):
    """ Paarameters that are used to trigger the deployment"""
    min_r2 : float

@step
def deployment_trigger(
    r2:float,
    params: DeploymentTriggerParameters)->bool:
    """simmple model deployment trigger that looks at the input model rmse and decides if it is good enough to deploy"""
    return r2 > params.min_r2

