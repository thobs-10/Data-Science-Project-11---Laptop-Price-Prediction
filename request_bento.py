import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import requests
import json
from typing import Tuple
import logging

SERVICE_URL = "http://localhost:3000/predict"

def make_inference(
    input_data:np.ndarray,
    service_url:str) -> str:
    '''make inference from the api'''
    serialized_input_data = json.dumps(input_data.tolist())
    response = requests.post(
        service_url,
        data=serialized_input_data,
        headers={"content-type":"application/json"}
    )
    return response.text

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
        y_test = pd.read_csv("C:\\Users\\Cash Crusaders\\Desktop\\My Portfolio\\Projects\\Data Science Projects\\Data Science Project 11 - Laptop Price Prediction\\dataset\\feature_engineered_data\\y_test_df.csv")
        
        X_train.drop(columns=['Unnamed: 0'],inplace=True)
        X_train.drop(columns=['Cpu'],inplace=True)
        X_train.drop(columns=['Gpu'],inplace=True)
        X_train.drop(columns=['Cpu Name'],inplace=True)
        #for testing
        # X_test_copy = X_test.iloc[:10]
        # y_test_copy = y_test.iloc[:10]
        X_test.drop(columns=['Unnamed: 0'],inplace=True)
        X_test.drop(columns=['Cpu'],inplace=True)
        X_test.drop(columns=['Gpu'],inplace=True)
        X_test.drop(columns=['Cpu Name'],inplace=True)

        y_test.drop(columns=['Unnamed: 0'],inplace=True)

        return X_test, y_test, X_train
    
def inference_loader():
    """
    Args:
        None
    Returns:
        df: pd.DataFrame
    """
    try:
        ingest_data = IngestData()
        X_test, y_test, X_train = ingest_data.get_data()
        return X_test, y_test, X_train
    except Exception as e:
        logging.error(e)
        raise e
    
def outcome(pred):
    print(f"{pred}")

def main():
    X_test, y_test, X_train = inference_loader()
    column_transformer = ColumnTransformer(transformers=[
            ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
        ],remainder='passthrough')
    X_train_tranformed = np.array(column_transformer.fit_transform(X_train))
    input_data_tranformed = np.array(column_transformer.transform(X_test))
    # X_test_array = np.array(X_test)
    # X_test_array.reshape(1,12)
    preds = make_inference(input_data_tranformed,SERVICE_URL)
    outcome(preds)

if __name__ == "__main__":
    main()