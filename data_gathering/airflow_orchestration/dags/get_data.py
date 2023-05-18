# get data from the dataset folder 
import pandas as pd
import numpy as np
import logging

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

def ingest_data():
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

