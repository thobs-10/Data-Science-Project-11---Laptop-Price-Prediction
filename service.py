import numpy as np
import bentoml
from bentoml.io import NumpyNdarray

laptop_model_runner = bentoml.sklearn.get("laptop-model:latest").to_runner()

svc = bentoml.Service("laptop_reg_service", runners=[laptop_model_runner])

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def predict(input_series: np.ndarray) -> np.ndarray:
    result = laptop_model_runner.predict.run(input_series)
    return result