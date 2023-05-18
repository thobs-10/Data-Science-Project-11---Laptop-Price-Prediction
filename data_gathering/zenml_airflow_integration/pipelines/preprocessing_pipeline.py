import pandas as pd
import numpy as np

from zenml.steps import step, Output

from zenml.integrations.airflow.flavors.airflow_orchestrator_flavor import AirflowOrchestratorSettings
from zenml.integrations.airflow.orchestrators.airflow_orchestrator import AirflowOrchestrator
from zenml.integrations.airflow.flavors.airflow_orchestrator_flavor import AirflowOrchestratorFlavor

from zenml.config import DockerSettings
from zenml.integrations.constants import AIRFLOW
flavour = AirflowOrchestratorFlavor()

docker_settings = DockerSettings(
    required_integrations = [AIRFLOW],
    requirements=["apache-airflow-providers-docker"]
)


airflow_settings = AirflowOrchestratorSettings(
    operator="docker",  # or "kubernetes_pod"
    
    # Dictionary of arguments to pass to the operator __init__ method
    operator_args={}
    )

from zenml.config import DockerSettings
from zenml.integrations.constants import AIRFLOW
from zenml.pipelines import pipeline

@pipeline(enable_cache=True, enable_artifact_metadata=True, settings={"orchestrator.airflow": airflow_settings, "docker":docker_settings})
def preprocessing_workflow(
    ingest_data,
    fix_column_datatypes,
    create_column,
    preprocess_features,
    drop_unwanted_columns,
    apply_fetch_processors
) -> Output(
    preprop_output = pd.DataFrame
):
     
    # the steps for the DAG 
    # get data from get_data script
    #get_data()
    df = ingest_data()
    df = fix_column_datatypes(df)
    df = create_column(df)
    df = preprocess_features(df)
    df = drop_unwanted_columns(df)
    df = apply_fetch_processors(df)
    return df