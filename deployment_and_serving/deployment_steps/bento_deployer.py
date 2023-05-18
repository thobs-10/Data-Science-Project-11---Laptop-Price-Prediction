from zenml.integrations.bentoml.steps import (
    BentoMLDeployerParameters,
    bentoml_model_deployer_step,
)

MODEL_NAME = 'laptop-price-model'

bentoml_model_deployer = bentoml_model_deployer_step(
    params=BentoMLDeployerParameters(
        model_name=MODEL_NAME,
        port=3001,
        production=False,
    )
)