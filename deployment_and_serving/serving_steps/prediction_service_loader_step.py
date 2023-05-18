import json
import os

import numpy as np
import pandas as pd
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step

from zenml.pipelines import pipeline
from zenml.steps import step, Output, BaseParameters

model_deployer = mlflow_model_deployer_step

class MLFlowDeploymentLoaderStepParameters(BaseParameters):
    """MLflow deployment getter parameters
    Attributes:
        pipeline_name: name of the pipeline that deployed the MLflow prediction
            server
        step_name: the name of the step that deployed the MLflow prediction
            server
        running: when this flag is set, the step only returns a running service
        model_name: the name of the model that is deployed
    """

    pipeline_name: str
    pipeline_step_name: str
    running : True
    model_name: str
    

# prediction_service_loader
@step(enable_cache=False, name='deployment')
def prediction_service_loader(
    params: MLFlowDeploymentLoaderStepParameters,
) -> MLFlowDeploymentService:
    """Get the prediction service started by the deployment pipeline"""

    # get the MLflow model deployer stack component
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    # fetch existing services with same pipeline name, step name and model name
    existing_services = model_deployer.find_model_server(
        pipeline_name=params.pipeline_name,
        pipeline_step_name=params.pipeline_step_name,
        model_name = params.model_name,
        running = params.running
    )

    if not existing_services:
        raise RuntimeError(
            f"No MLflow prediction service deployed by the "
            f"{params.step_name} step in the {params.pipeline_name} "
            f"pipeline is currently "
            f"running."
        )

    return existing_services[0]
