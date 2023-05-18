import os
import numpy as np
import pandas as pd
from datetime import datetime

import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from zenml.client import Client
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.experiment_trackers import (
    MLFlowExperimentTracker,
)
from zenml.steps import step, Output, step_output
# get the stack experiment tracking uri
# MLFLOW_TRACKING_URI = get_tracking_uri()

#get the experiment tracker of the active stack from the stack components
experiment_tracker = Client().active_stack.experiment_tracker
# the backend trcking server
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
# MLFLOW_TRACKING_URI = "./mlruns"
MODEL_NAME = 'laptop-price-model'
#TUNING_EXPERIMENT_NAME='hyper-optimization-experiment-1'
# set the stack experiment tracking uri to be the current working one
mlflow.set_tracking_uri("sqlite:///laptop.db")
# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

@step(enable_cache = True)
def model_registering(TUNING_EXPERIMENT_NAME:str)->Output(
    model_name = str
):
    """ this function is responsible for taking the model in tuning experiment to model registry
    is the sqlite database used by the tracking server to track the experiments
    """
    tuning_experiment_name = TUNING_EXPERIMENT_NAME
    # interacting with the mlflow beckend tracker\ tracking server
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    # client.get_experiment(experiment_id=2)
    prev_experiment = client.get_experiment_by_name(tuning_experiment_name)
    best_run = client.search_runs(
    experiment_ids=prev_experiment.experiment_id,
    filter_string="metrics.rmse < 3",
    run_view_type=ViewType.ACTIVE_ONLY,
    max_results=2,
    order_by=["metrics.rmse ASC"])[1]

    # interacting with model registry
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    model_uri = f"runs:/{best_run.info.run_id}/model"
    mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
    
    latest_versions = client.get_latest_versions(name=MODEL_NAME)
    for version in latest_versions:
        model_version = version.version
        
    # registering the fresh new model into staging
    new_stage = "Staging"
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=model_version,
        stage=new_stage,
        archive_existing_versions=False
    )
    
    # update the mlflow client
    date = datetime.today().date()
    client.update_model_version(
        name=MODEL_NAME,
        version=model_version,
        description=f"The model version {model_version} was transitioned to {new_stage} on {date}"
    )
    return MODEL_NAME
        

