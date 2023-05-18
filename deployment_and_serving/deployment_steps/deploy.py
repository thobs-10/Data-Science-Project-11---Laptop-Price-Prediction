from typing import Optional, Type, cast
import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient, artifact_utils

from zenml.client import Client
# from zenml.constants import (
#     DEFAULT_SERVICE_START_STOP_TIMEOUT,
#     MLFLOW_MODEL_FORMAT,
# )
from zenml.environment import Environment
from zenml.integrations.mlflow.experiment_trackers.mlflow_experiment_tracker import (
    MLFlowExperimentTracker,
)
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services.mlflow_deployment import (
    MLFlowDeploymentConfig,
    MLFlowDeploymentService,
)
from zenml.logger import get_logger
from zenml.materializers import UnmaterializedArtifact

from zenml.steps import (
    STEP_ENVIRONMENT_NAME,
    BaseParameters,
    BaseStep,
    StepEnvironment,
)
from zenml.steps import step, Output

mlflow.set_tracking_uri("sqlite:///laptop.db")
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
DEFAULT_SERVICE_START_STOP_TIMEOUT = 60

run_name = "wistful-croc-520"
run_id = "535f15be557f49bf8de82caede125417"
experiment_name = "hyper-optimization-experiment-1"
model_name = "laptop-price-model"

logger = get_logger(__name__)

class MLFlowDeployerParameters(BaseParameters):
    """Model deployer step parameters for MLflow.
    Attributes:
        model_name: the name of the MLflow model logged in the MLflow artifact
            store for the current pipeline.
        experiment_name: Name of the MLflow experiment in which the model was
            logged.
        run_name: Name of the MLflow run in which the model was logged.
        workers: number of workers to use for the prediction service
        mlserver: set to True to use the MLflow MLServer backend (see
            https://github.com/SeldonIO/MLServer). If False, the
            MLflow built-in scoring server will be used.
        registry_model_name: the name of the model in the model registry
        registry_model_version: the version of the model in the model registry
        registry_model_stage: the stage of the model in the model registry
        replace_existing: whether to create a new deployment service or not,
            this parameter is only used when trying to deploy a model that
            is registered in the MLflow model registry. Default is True.
        timeout: the number of seconds to wait for the service to start/stop.
    """

    model_name: str = "model"
    registry_model_name: Optional[str] = None
    registry_model_version: Optional[str] = None
    # registry_model_stage: Optional[ModelVersionStage] = None
    experiment_name: Optional[str] = None
    run_name: Optional[str] = None
    replace_existing: bool = True
    workers: int = 1
    mlserver: bool = False
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT


@step(enable_cache=False,name='deployment_step')
def deploy_model_step(
    deploy_decision: bool,
    model: UnmaterializedArtifact,
    params: MLFlowDeployerParameters,
    ) -> MLFlowDeploymentService:
    """Model deployer pipeline step for MLflow.
    This step deploys a model logged in the MLflow artifact store to a
    deployment service. The user would typically use this step in a pipeline
    that deploys a model that was already registered in the MLflow model
    registr either manually or by using the `mlflow_model_registry_step`.
    Args:
        deploy_decision: whether to deploy the model or not
        model: the model artifact to deploy
        params: parameters for the deployer step
    Returns:
        MLflow deployment service
    Raises:
        ValueError: if the MLflow experiment tracker is not found
    """
    model_deployer = cast(
        MLFlowModelDeployer, MLFlowModelDeployer.get_active_model_deployer()
    )
    
    experiment_tracker = Client().active_stack.experiment_tracker
    if not isinstance(experiment_tracker, MLFlowExperimentTracker):
        raise ValueError(
            "MLflow model deployer step requires an MLflow experiment "
            "tracker. Please add an MLflow experiment tracker to your "
            "stack."
        )

    # get pipeline name, step name and run id
    step_env = cast(StepEnvironment, Environment()[STEP_ENVIRONMENT_NAME])
    pipeline_name = step_env.pipeline_name
    run_name = step_env.run_name
    step_name = step_env.step_name

    # Configure Mlflow so the client points to the correct store
    experiment_tracker.configure_mlflow()
    # configure in my own way on  how to access the model registry
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    # get experiment name or pipeline name
    
    # get run_name from the mlflow
    
    # use the experiment name and run name to get the mlflow run id
    mlflow_run_id = run_id
    # mlflow_run_id = experiment_tracker.get_run_id(
    #     experiment_name=params.experiment_name or pipeline_name,
    #     run_name=params.run_name or run_name,
        
    # )

    model_uri = ""
    if mlflow_run_id and client.list_artifacts(
        mlflow_run_id, params.model_name
    ):
        model_uri = artifact_utils.get_artifact_uri(
            run_id=mlflow_run_id, artifact_path=params.model_name,
            tracking_uri= MLFLOW_TRACKING_URI
        )

    # fetch existing services with same pipeline name, step name and model name
    existing_services = model_deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=step_name,
        model_name=params.model_name,
    )
    

    # create a config for the new model service
    predictor_cfg = MLFlowDeploymentConfig(
        model_name=params.model_name or "",
        model_uri=model_uri,
        workers=params.workers,
        mlserver=params.mlserver,
        registry_model_name=params.registry_model_name or "",
        registry_model_version=params.registry_model_version or "",
        pipeline_name=pipeline_name,
        run_name=run_name,
        pipeline_run_id=run_name,
        pipeline_step_name=step_name,
        timeout=params.timeout,
    )

     # Creating a new service with inactive state and status by default
    service = MLFlowDeploymentService(predictor_cfg)
    if existing_services:
        service = cast(MLFlowDeploymentService, existing_services[0])

    # check for conditions to deploy the model
    if not model_uri:
        # an MLflow model was not trained in the current run, so we simply reuse
        # the currently running service created for the same model, if any
        if not existing_services:
            logger.warning(
                f"An MLflow model with name `{params.model_name}` was not "
                f"logged in the current pipeline run and no running MLflow "
                f"model server was found. Please ensure that your pipeline "
                f"includes a step with a MLflow experiment configured that "
                "trains a model and logs it to MLflow. This could also happen "
                "if the current pipeline run did not log an MLflow model  "
                f"because the training step was cached."
            )
            # return an inactive service just because we have to return
            # something
            return service
        logger.info(
            f"An MLflow model with name `{params.model_name}` was not "
            f"trained in the current pipeline run. Reusing the existing "
            f"MLflow model server."
        )
        if not service.is_running:
            service.start(params.timeout)
            logger.info(
            f"MLflow deployment service started and reachable at:\n"
            f" prediction url for request:    {service.prediction_url}\n"
            f" service endpoint:  {service.endpoint}\n"
    )
        # return the existing service
        return service

    # even when the deploy decision is negative, if an existing model server
    # is not running for this pipeline/step, we still have to serve the
    # current model, to ensure that a model server is available at all times
    if not deploy_decision and existing_services:
        logger.info(
            f"Skipping model deployment because the model quality does not "
            f"meet the criteria. Reusing last model server deployed by step "
            f"'{step_name}' and pipeline '{pipeline_name}' for model "
            f"'{params.model_name}'..."
        )
        # even when the deploy decision is negative, we still need to start
        # the previous model server if it is no longer running, to ensure
        # that a model server is available at all times
        if not service.is_running:
            service.start(params.timeout)
        return service
    
    # create a new model deployment and replace an old one if it exists
    new_service = cast(
        MLFlowDeploymentService,
        model_deployer.deploy_model(
            replace=True,
            config=predictor_cfg,
            timeout=params.timeout,
        ),
    )

    logger.info(
        f"MLflow deployment service started and reachable at:\n"
        f"    {new_service.prediction_url}\n"
    )

    return new_service