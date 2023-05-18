import pandas as pd
import numpy as np

from zenml.integrations.deepchecks.steps import (
    DeepchecksDataDriftCheckStepParameters,
    deepchecks_data_drift_check_step,
)

LABEL_COL = "Price"
	   	
data_drift_detector = deepchecks_data_drift_check_step(
    step_name="data_drift_detector",
    params=DeepchecksDataDriftCheckStepParameters(
        dataset_kwargs=dict(cat_features=['Company','TypeName','Cpu brand','Gpu brand','os']),
    ),
)

