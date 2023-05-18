import json
import os

import numpy as np
import pandas as pd
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService

from zenml.integrations.mlflow.steps.mlflow_deployer import mlflow_model_deployer_step
from deployment_and_serving.deployment_steps.deploy import deploy_model_step

from zenml.integrations.mlflow.flavors.mlflow_model_deployer_flavor import MLFlowModelDeployerFlavor
from zenml.pipelines import pipeline
from zenml.steps import step, Output, BaseParameters

model_deployer = mlflow_model_deployer_step
# model_deployer = MLFlowModelDeployer.get_active_model_deployer()

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
    step_name: str
    model_name: str
    running : bool

# prediction_service_loader
@step(enable_cache=False, name='deployment_step')
def deployment(
    params: MLFlowDeploymentLoaderStepParameters,
) -> MLFlowDeploymentService:
    """Get the prediction service started by the deployment pipeline"""

    # get the MLflow model deployer stack component
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    # fetch existing services with same pipeline name, step name and model name
    existing_services = model_deployer.find_model_server(
        pipeline_name=params.pipeline_name,
        pipeline_step_name=params.step_name,
    )

    if not existing_services:
        raise RuntimeError(
            f"No MLflow prediction service deployed by the "
            f"{params.step_name} step in the {params.pipeline_name} "
            f"pipeline is currently "
            f"running."
        )

    return existing_services[0]




        
        