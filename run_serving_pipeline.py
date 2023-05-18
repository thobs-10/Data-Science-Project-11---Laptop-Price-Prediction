from deployment_and_serving.serving_steps import get_user_input
from deployment_and_serving.serving_steps import inference_preprocessing
from deployment_and_serving.serving_steps import prediction_service_loader_step
from deployment_and_serving.serving_steps import predictor_step

from deployment_and_serving.serving_pipeline import serving_pipeline

from deployment_and_serving.serving_steps.prediction_service_loader_step import MLFlowDeploymentLoaderStepParameters

from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)

def serving_main():
    serving = serving_pipeline(
        dynamic_data=get_user_input(),
        prediction_service_loader=prediction_service_loader_step(
            MLFlowDeploymentLoaderStepParameters(
                pipeline_name="continuous_deployment_pipeline",
                step_name="model_deployer",
            )
        ),
        predictor=predictor_step(),
    )
    serving.run()

    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri {get_tracking_uri()}\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "You can find your runs tracked within the `mlflow_example_pipeline`"
        "experiment. Here you'll also be able to compare the two runs.)"
    )

    model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    # fetch existing services with same pipeline name, step name and model name
    service = model_deployer.find_model_server(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="model_deployer",
        running=True,
    )

    if service[0]:
        print(
            f"The MLflow prediction server is running locally as a daemon process "
            f"and accepts inference requests at:\n"
            f"    {service[0].prediction_url}\n"
            f"To stop the service, re-run the same command and supply the "
            f"`--stop-service` argument."
        )

if __name__ == "__main__":
    serving_main()