from typing import cast

import click
from rich import print
# steps for the continous deployment pipeline 
from deployment_steps.deployment_trigger import (
    DeploymentTriggerParameters,
    deployment_trigger
)
from deployment_steps.data_loader import ingest_data
from deployment_steps.evaluator import model_evaluator
from deployment_steps.deployment_trigger import deployment_trigger
from deployment_steps.deployment import mlflow_model_deployer_step
from deployment_steps.deployment import deployment
# from deployment_steps.predictor import predictor
from deployment_steps.deployment import (
    MLFlowDeploymentLoaderStepParameters,
    model_deployer,
    deployment
)

#steps for the serving pipeline
from serving_steps.get_user_input import get_input
from serving_steps.inference_preprocessing import preprocess_input
from serving_steps.prediction_service_loader_step import (
    prediction_service_loader,
)
from serving_steps.predictor_step import predictor

from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer
)

from deployment_pipeline.deployment_pipeline import continous_deployment
from serving_pipeline.serving_pipeline import inference_pipeline

from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import MLFlowDeployerParameters

DEPLOY = "deploy"
PREDICT = "predict"
DEPLOY_AND_PREDICT = "deploy_and_predict"

service_port = ""
# for inference pipeline
pipeline_name="deployment_pipeline"
step_name="deployment_step"
model_name="laptop-price-model"
running =True

@click.command()
@click.option(
    "--config",
    "--c",
    type = click.Choice([DEPLOY,PREDICT,DEPLOY_AND_PREDICT]),
    default = DEPLOY_AND_PREDICT,
    help = "Optionally you can choose to only rn the deployment pipeline to train and deploy a model(deploy)."
    "or to run a predictioon against the deployed model(predict)"
    "by default this system runs both (deploy and predict)"
)
@click.option(
    "--min-r2",
    default = 0.80,
    help = "minimum accuracy required to deploy the model",
)

def main(config:str,min_r2:float):
    """run the MLFlow project pipeline"""

    # get the mlflow model deployer stack component
    mlflow_model_deployer_component = (
        MLFlowModelDeployer.get_active_model_deployer()
    )

    deploy = config == DEPLOY or config == DEPLOY_AND_PREDICT
    predict = config == PREDICT or config == DEPLOY_AND_PREDICT

    if deploy:
        #initiialize a continuous deployment pipeline run
        continuous_deployment_pipeline = continous_deployment(
            ingest_data(),
            model_evaluator(),
            deployment_trigger = deployment_trigger(
            params = DeploymentTriggerParameters(
            min_r2=min_r2
            )
            ),
            model_deployer = deployment(
            params = MLFlowDeployerParameters(workers=3,
                                              timeout=30)
            ),
            
        )

        continuous_deployment_pipeline.run()

    if predict:
        # initialize serving pipeline
        serving_inference_pipeline = inference_pipeline(
            get_input(),
            preprocess_input(),
            prediction_service_loader = prediction_service_loader(
            MLFlowDeploymentLoaderStepParameters(
            pipeline_name="continous_deployment",
            pipeline_step_name = "model_deployer",
            running=False,
            model_name=model_name
            )
            ),
            predictor = predictor(),
        )
        serving_inference_pipeline.run()

    print(" You can run:\n"
          f"[itallic green]  mlflow ui --backend-store-uri '{get_tracking_uri()}"
          "[/italic green]\n ...to inspect your experimen run within mlflow"
          "UI.\nYou can find your runs tracked within the "
          "'project name' experiment.")
    
    # fetch the existing service with the samee pipeline name, step name and model name
    existing_service = mlflow_model_deployer_component.find_model_server(
        pipeline_name="continous_deployment",
        pipeline_step_name="model_deployer",
        model_name="laptop-price-model"
    )

    if existing_service:
        service = cast(MLFlowDeploymentService, existing_service[0])
        if service.is_running:
            print("The MLFlow prediction server is running locally as a deamon process"
                  f" seervice and accepts inference request at:\n"
                  f" {service.prediction_url}\n"
                  f"To stop the service, run "
                  f"[italic green]'zenml model-deployer models delete "
                  f"{str(service.uuid)}'[/italic green].")
            
        elif service.is_failed:
            print(
                f"The MLFlow prediction server is in a failed state:\n"
                f"Last state: '{service.status.state.value}'\n"
                f"Last Error: '{service.status.last_error}'"
            )
    else:
        print(
            "No MLFlow prediction server is currently running. The deployment pipeline must run first to deploy the model."
            "Execute the same command with '--deploy' argument to deploy a model"
        )
    


if __name__=="__main__":
    main()


