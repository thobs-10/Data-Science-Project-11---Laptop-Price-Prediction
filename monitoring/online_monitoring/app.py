import os
import pickle
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


from flask import Flask
from flask import request
from flask import jsonify
import requests
import json
from typing import Tuple
import logging
# for the mangoDb database
from pymongo import MongoClient
SERVICE_URL = "http://localhost:3000/predict"
# to access the predictive model that has the dictvevtorizer and moddel
MODEL_FILE = os.getenv('MODEL_FILE', 'rf_reg.pkl')
# the address where the data will be sent for monitoring using evidently 
EVIDENTLY_SERVICE_ADDRESS = os.getenv('EVIDENTLY_SERVICE', 'http://127.0.0.1:5000')
# MANGODB for storing the data before it goes to the premotheous and evidently or monitoring service
MONGODB_ADDRESS = os.getenv("MONGODB_ADDRESS", "mongodb://127.0.0.1:27017")

with open(MODEL_FILE, 'rb') as f_in:
    model = pickle.load(f_in)

# creating a flask
app = Flask('laptop-price')
# creating the database
mongo_client = MongoClient(MONGODB_ADDRESS)
# getting the database and calling the predictive service
db = mongo_client.get_database("prediction_service")
# getting the data that will be stored and sent to evidently
collection = db.get_collection("data")

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
        y_train = pd.read_csv("C:\\Users\\Cash Crusaders\\Desktop\\My Portfolio\\Projects\\Data Science Projects\\Data Science Project 11 - Laptop Price Prediction\\dataset\\feature_engineered_data\\y_train_df.csv")

        X_train.drop(columns=['Unnamed: 0'],inplace=True)
        X_train.drop(columns=['Cpu'],inplace=True)
        X_train.drop(columns=['Gpu'],inplace=True)
        X_train.drop(columns=['Cpu Name'],inplace=True)

        y_train.drop(columns=['Unnamed: 0'],inplace=True)
        #for testing
        # X_test_copy = X_test.iloc[:10]
        # y_test_copy = y_test.iloc[:10]
        X_test.drop(columns=['Unnamed: 0'],inplace=True)
        X_test.drop(columns=['Cpu'],inplace=True)
        X_test.drop(columns=['Gpu'],inplace=True)
        X_test.drop(columns=['Cpu Name'],inplace=True)

        y_test.drop(columns=['Unnamed: 0'],inplace=True)

        return X_test, y_test, X_train, y_train
    
def inference_loader():
    """
    Args:
        None
    Returns:
        df: pd.DataFrame
    """
    try:
        ingest_data = IngestData()
        X_test, y_test, X_train, y_train = ingest_data.get_data()
        return X_test, y_test, X_train, y_train
    except Exception as e:
        logging.error(e)
        raise e


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

@app.route('/predict', methods=['POST'])
def predict():
    # getting the json file that has the records
    record = request.get_json()
    # transform the record data
    #record_dict = json.loads(record)
    record_df = pd.DataFrame(record['data'],
                  columns=['Company', 'TypeName', 'Ram', 'Weight','Touchscreen','Ips','ppi','Cpu brand','HDD','SSD','Gpu brand','os'])

    numpy_record = np.array(record['data'])
    # numpy_record = np.array(record)
    # print(record)
    # get the data to besu
    # X_test, y_test, X_train, y_train = inference_loader()
    
    # column_transformer = ColumnTransformer(transformers=[
    #         ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
    #     ],remainder='passthrough')
    # X_train_tranformed = np.array(column_transformer.fit_transform(X_train))
    # input_data_tranformed = np.array(column_transformer.transform(X_test))
    # X_test_array = np.array(X_test)
    # X_test_array.reshape(1,12)
    # preds = make_inference(input_data_tranformed,SERVICE_URL)
    preds = model.predict(numpy_record)
    df_pred = pd.DataFrame()
    df_pred['predicted_price'] = preds
    pred_json  = df_pred.to_json()
    # result = {
    #     'predicted_price':pred_json
    # }

    # create a function that will save the data predicted to the database
    record_df['prediction'] = preds
    record_df.to_parquet('C:\\Users\\Cash Crusaders\\Desktop\\My Portfolio\\Projects\\Data Science Projects\\Data Science Project 11 - Laptop Price Prediction\\monitoring\\online_monitoring\\collection_db.parquet')
    rec_dict = record_df.to_dict('records')
    # save_to_db(record_df, preds)
    # a function to send the data to evidently service
    # send_to_evidently_service(record_df, preds)
    # return tthe results in A jsonify file
    return jsonify(rec_dict)

# function that takse in the records and teh predicted results, store them in db
def save_to_db(record, prediction):
    # create a copy of the record
    rec = record.copy()
    #in the records dataframe,create a column  for the predicted values
    rec['prediction'] = prediction
    rec.to_parquet('C:\\Users\\Cash Crusaders\\Desktop\\My Portfolio\\Projects\\Data Science Projects\\Data Science Project 11 - Laptop Price Prediction\\monitoring\\online_monitoring\\collection_db.parquet')
    # convert ddataframe into a dictionary
    # rec_dict = rec.to_dict('records')
    # add the recods to database
    # collection.insert_many(rec_dict)

# function to get the records and the predicted records to the monitoring service
def send_to_evidently_service(record, prediction):
    # make a copy of the records
    rec = record.copy()
    # create a column of the predicted values
    rec['prediction'] = prediction
    # rec_dict = rec.to_dict('records')
    # convert the df to json
    rec_json = rec.to_json()
    # qwe neeed ri post the records in a json file to the evidently service address
    requests.post(f"{EVIDENTLY_SERVICE_ADDRESS}/iterate/taxi", json=rec_json)

# def main():
#     results = endpoint_prediction()

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)