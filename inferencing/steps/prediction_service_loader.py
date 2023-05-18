from typing import cast

from zenml.integrations.bentoml.model_deployers.bentoml_model_deployer import (
    BentoMLModelDeployer,
)
from zenml.integrations.bentoml.services.bentoml_deployment import (
    BentoMLDeploymentService,
)
from zenml.steps import BaseParameters, step

class PredictionServiceLoaderStepParameters(BaseParameters):
    """Prediction Service loader parameters.

    Attrs:
        pipeline_name: name of the pipeline that deployed the model.
        step_name: the name of the step that deployed the model.
        model_name: the name of the model that was deployed.
    """

    pipeline_name: str
    step_name: str
    model_name: str


@step(enable_cache=False)
def bentoml_prediction_service_loader(
    params: PredictionServiceLoaderStepParameters,
) -> BentoMLDeploymentService:
    """Get the BentoML prediction service started by the deployment pipeline."""
    model_deployer = BentoMLModelDeployer.get_active_model_deployer()

    services = model_deployer.find_model_server(
        pipeline_name=params.pipeline_name,
        pipeline_step_name=params.step_name,
        model_name=params.model_name,
    )
    if not services:
        raise RuntimeError(
            f"No BentoML prediction server deployed by the "
            f"'{params.step_name}' step in the '{params.pipeline_name}' "
            f"pipeline for the '{params.model_name}' model is currently "
            f"running."
        )

    if not services[0].is_running:
        raise RuntimeError(
            f"The BentoML prediction server last deployed by the "
            f"'{params.step_name}' step in the '{params.pipeline_name}' "
            f"pipeline for the '{params.model_name}' model is not currently "
            f"running."
        )

    return cast(BentoMLDeploymentService, services[0])