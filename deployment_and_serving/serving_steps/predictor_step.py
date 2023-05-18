import pandas as pd
import json
import numpy as np
from zenml.steps import step
from zenml.integrations.mlflow.services import MLFlowDeploymentService


@step()
def predictor(
    service: MLFlowDeploymentService,
    data: str,
) -> np.ndarray:
    """Run an inference request against a prediction service"""

    service.start(timeout=10)  # should be a NOP if already started
    data = json.loads(data)
    data.pop("columns")
    data.pop("index")
    columns_for_df = [
        "Company",
        "TypeName",
        "Ram",
        "Weight",
        "Touchscreen",
        "Ips",
        "ppi",
        "Cpu",
        "brand",
        "HDD",
        "SSD",
        "Gpu",
        "brand",
        "os",
    ]
    df = pd.DataFrame(data["data"], columns=columns_for_df)
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data = np.array(json_list)
    prediction = service.predict(data)
    return prediction