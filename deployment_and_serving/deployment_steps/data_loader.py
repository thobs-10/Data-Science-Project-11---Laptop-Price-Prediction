import numpy as np
import pandas  as pd

# get data from the dataset folder 
import pandas as pd
import numpy as np
import logging
from zenml.steps import Output, step

from zenml.integrations.airflow.flavors.airflow_orchestrator_flavor import AirflowOrchestratorSettings

from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW

docker_settings = DockerSettings(required_integrations=[MLFLOW])
class IngestData:
    """
    Data ingestion class which ingests data from the source and returns a DataFrame.
    """

    def __init__(self) -> None:
        """Initialize the data ingestion class."""
        pass

    def get_data(self):
        X_train = pd.read_parquet("C:\\Users\\Cash Crusaders\\Desktop\\My Portfolio\\Projects\\Data Science Projects\\Data Science Project 11 - Laptop Price Prediction\\dataset\\feature_engineered_data\\X_train_df.parquet.gzip")
        X_test = pd.read_parquet("C:\\Users\\Cash Crusaders\\Desktop\\My Portfolio\\Projects\\Data Science Projects\\Data Science Project 11 - Laptop Price Prediction\\dataset\\feature_engineered_data\\X_test_df.parquet")
        y_train = pd.read_csv("C:\\Users\\Cash Crusaders\\Desktop\\My Portfolio\\Projects\\Data Science Projects\\Data Science Project 11 - Laptop Price Prediction\\dataset\\feature_engineered_data\\y_train_df.csv")
        y_test = pd.read_csv("C:\\Users\\Cash Crusaders\\Desktop\\My Portfolio\\Projects\\Data Science Projects\\Data Science Project 11 - Laptop Price Prediction\\dataset\\feature_engineered_data\\y_test_df.csv")
        # for training
        X_train.drop(columns=['Unnamed: 0'],inplace=True)
        X_train.drop(columns=['Cpu'],inplace=True)
        X_train.drop(columns=['Gpu'],inplace=True)
        X_train.drop(columns=['Cpu Name'],inplace=True)
        y_train.drop(columns=['Unnamed: 0'],inplace=True)
        #for testing
        X_test.drop(columns=['Unnamed: 0'],inplace=True)
        X_test.drop(columns=['Cpu'],inplace=True)
        X_test.drop(columns=['Gpu'],inplace=True)
        X_test.drop(columns=['Cpu Name'],inplace=True)
        y_test.drop(columns=['Unnamed: 0'],inplace=True)
        return X_train,X_test,y_test,y_train
    
@step
def ingest_data()->Output(
        X_train = pd.DataFrame,
        X_test = pd.DataFrame,
        y_test =pd.DataFrame,
        y_train = pd.DataFrame
):
    """
    Args:
        None
    Returns:
        df: pd.DataFrame
    """
    try:
        ingest_data = IngestData()
        X_train,X_test,y_test_series,y_train_series = ingest_data.get_data()
        return X_train,X_test,y_test_series,y_train_series
    except Exception as e:
        logging.error(e)
        raise e

