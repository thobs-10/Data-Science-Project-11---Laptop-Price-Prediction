from zenml.integrations.bentoml.steps import (
    BentoMLBuilderParameters,
    bento_builder_step,
)

MODEL_NAME = 'laptop-price-model'

bento_builder = bento_builder_step(
    params=BentoMLBuilderParameters(
        model_name=MODEL_NAME,
        model_type="scikit-learn",
        service= "service:svc",
        labels={
            "framework": "scikit-learn",
            # "dataset": "mnist",
            "zenml_version": "0.21.1",
        },
        exclude=["data"],
        python={
            "packages": ["zenml", "scikit-learn"],
        },
    )
)