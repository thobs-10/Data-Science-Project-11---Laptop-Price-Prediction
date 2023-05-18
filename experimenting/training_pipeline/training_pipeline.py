import pandas as pd
import numpy as np

from zenml.config import DockerSettings
from zenml.steps import step, Output
from zenml.integrations.constants import DEEPCHECKS, SKLEARN
from zenml.pipelines import pipeline

docker_settings = DockerSettings(required_integrations=[DEEPCHECKS, SKLEARN])

@pipeline(enable_cache=False ,enable_artifact_metadata=False,  settings={"docker":docker_settings})
def training_pipeline(
    ingest_data,
    train_model,
    fine_tune_model,
    model_registering,
    data_drift_validator,
):
    
    # the steps for the DAG 
    # get data from get_data script
    #get_data()
    X_train,X_test,y_test_series,y_train_series = ingest_data()
    TRAINING_EXPERIMENT_NAME = train_model(X_train=X_train, y_train= y_train_series, X_val= X_test, y_val= y_test_series)
    TUNING_EXPERIMENT_NAME = fine_tune_model(TRAINING_EXPERIMENT_NAME,X_train, y_train_series, X_test, y_test_series)
    MODEL_NAME = model_registering(TUNING_EXPERIMENT_NAME)
    data_drift_validator(reference_dataset = X_train, target_dataset = X_test)
    

    