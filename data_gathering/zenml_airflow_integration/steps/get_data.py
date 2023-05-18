# get data from the dataset folder 
import pandas as pd
import numpy as np
import logging
from zenml.steps import Output, step

from zenml.integrations.airflow.flavors.airflow_orchestrator_flavor import AirflowOrchestratorSettings

from zenml.config import DockerSettings
from zenml.integrations.constants import AIRFLOW

docker_settings = DockerSettings(
    required_integrations = [AIRFLOW],
    requirements=["apache-airflow-providers-docker"]
)
airflow_settings = AirflowOrchestratorSettings(
    operator="docker",  # or "kubernetes_pod"
    # Dictionary of arguments to pass to the operator __init__ method
    operator_args={}
)

class IngestData:
    """
    Data ingestion class which ingests data from the source and returns a DataFrame.
    """

    def __init__(self) -> None:
        """Initialize the data ingestion class."""
        pass

    def get_data(self) -> pd.DataFrame:
        df = pd.read_csv("C:\\Users\\Cash Crusaders\\Desktop\\My Portfolio\\Projects\\Data Science Projects\\Data Science Project 11 - Laptop Price Prediction\\dataset\\laptop_data.csv")
        return df
    
@step(settings={"orchestrator.airflow": airflow_settings,"docker": docker_settings})
def ingest_data()->Output(
        df=pd.DataFrame
):
    """
    Args:
        None
    Returns:
        df: pd.DataFrame
    """
    try:
        ingest_data = IngestData()
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(e)
        raise e

