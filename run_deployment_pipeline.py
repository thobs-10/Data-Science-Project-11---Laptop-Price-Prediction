from deployment_and_serving.deployment_steps.data_loader import ingest_data
from deployment_and_serving.deployment_steps.evaluator import model_evaluator
from deployment_and_serving.deployment_steps.deployment_trigger import deployment_trigger
from deployment_and_serving.deployment_steps.deploy import deploy_model_step
from deployment_and_serving.deployment_steps.deployment import deployment
from deployment_and_serving.deployment_steps.deploy_demo import deploy_demo
from deployment_and_serving.deployment_steps.bento_builder import bento_builder
from deployment_and_serving.deployment_steps.bento_deployer import bentoml_model_deployer
# from deployment_steps.predictor import predictor

from deployment_and_serving.deployment_steps.deployment_trigger import (
    DeploymentTriggerParameters,
)
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from deployment_and_serving.deployment_steps.deployment import (
    MLFlowDeploymentLoaderStepParameters,
    model_deployer,
    
)


from deployment_and_serving.deployment_steps.deploy import (
    MLFlowDeployerParameters
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
#from zenml.integrations.mlflow.steps import MLFlowDeployerParameters

from deployment_and_serving.deployment_pipeline.deployment_pipeline import continous_deployment

run_name = "wistful-croc-520"
run_id = "535f15be557f49bf8de82caede125417"
experiment_name = "hyper-optimization-experiment-1"
model_name = "laptop-price-model"

pipeline_name = "deployment_pipeline"
step_name = "deployment_step"

def main():
    continuous_deployment_pipeline = continous_deployment(
        ingest_data(),
        model_evaluator(),
        deployment_trigger(
        params = DeploymentTriggerParameters(min_r2=0.80)),
        bento_builder=bento_builder,
        deployer=bentoml_model_deployer,
        # model_deployer = model_deployer()
        # params=MLFlowDeploymentLoaderStepParameters(pipeline_name=pipeline_name,step_name=step_name,running=True, model_name=model_name,workers=3, timeout=60)),
        # model_deployer = deploy_model_step(
        # params=MLFlowDeployerParameters(model_name=model_name)),
        )

    continuous_deployment_pipeline.run(run_name="deploymentV4.5")

if __name__ == "__main__":
    main()
