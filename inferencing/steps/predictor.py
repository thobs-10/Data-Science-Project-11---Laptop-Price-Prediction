from typing import Dict, List

import numpy as np
import  pandas as pd
from rich import print as rich_print

from zenml.integrations.bentoml.services import BentoMLDeploymentService
from zenml.steps import step


@step
def predictor(
    query_data: pd.DataFrame,
    service: BentoMLDeploymentService,
) -> None:
    """Run an inference request against the BentoML prediction service.

    Args:
        service: The BentoML service.
        data: The data to predict.
    """
    service.start(timeout=10)  # should be a NOP if already started
    prediction = service.predict("predict", np.array(query_data))
    rich_print(f"Prediction : {prediction}")
    return prediction
        