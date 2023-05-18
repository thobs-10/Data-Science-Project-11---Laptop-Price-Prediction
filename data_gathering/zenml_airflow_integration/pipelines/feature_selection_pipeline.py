import numpy as np
import pandas as pd

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
def feature_selection_workflow(
   seperate_dataset,
   label_encode,
   drop_top_correlated_features,
   drop_correlated,
   split_for_PCA,
   feature_scaling
):
     
    # the steps for the DAG 
    original_X,copy_X,y = seperate_dataset()
    ndarray_X = label_encode(original_X)
    to_drop, df_X =drop_top_correlated_features(copy_X)
    new_X_df = drop_correlated(to_drop, df_X)
    X_train,X_test, y_train, y_test = split_for_PCA(ndarray_X,y)
    X_train_scaled, X_test_scaled,y_train, y_test = feature_scaling( X_train, X_test, y_train, y_test)

