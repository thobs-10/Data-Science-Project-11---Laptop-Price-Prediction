import pandas as pd
import numpy as np

from zenml.steps import step, Output

from zenml.config import DockerSettings
from zenml.integrations.constants import AIRFLOW
from zenml.integrations.airflow.flavors.airflow_orchestrator_flavor import AirflowOrchestratorSettings
from zenml.pipelines import pipeline

docker_settings = DockerSettings(
    required_integrations = [AIRFLOW],
    requirements=["apache-airflow-providers-docker"]
)
airflow_settings = AirflowOrchestratorSettings(
    operator="docker",  # or "kubernetes_pod"
    
    # Dictionary of arguments to pass to the operator __init__ method
    operator_args={}
    )

# read dataframe 
# df = pd.read_parquet()
@pipeline(enable_cache=True, enable_artifact_metadata=True, settings={"orchestrator.airflow": airflow_settings, "docker":docker_settings})
def feature_engineering_workflow(
    memory_engineering,
    SDD_engineering,
    HDD_engineering,
    drop_columns,
    GPU_engineering,
    apply_cat_os,
    seperate_dataset,
    split_dataset,
    save_dataset
):
     
    # the steps for the DAG 
    df = memory_engineering()
    df = SDD_engineering(df)
    df = HDD_engineering(df)
    df = drop_columns(df)
    df = GPU_engineering(df)
    df =apply_cat_os(df)
    X, y = seperate_dataset(df)
    X_train,X_test, y_train, y_test = split_dataset(X,y)
    save_dataset(X_train,X_test, y_train, y_test)



   