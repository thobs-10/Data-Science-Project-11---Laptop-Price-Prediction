import os
import pickle

import requests
from flask import Flask
from flask import request
from flask import jsonify
# for the mangoDb database
from pymongo import MongoClient
from deployment_and_serving.run_deployment_serving_pipeline import main
from zenml.services import load_last_service_from_step


# to access the predictive model that has the dictvevtorizer and moddel
def get_model_as_service():
    service = load_last_service_from_step(
                pipeline_name="continuous_deployment_pipeline",
                step_name="model_deployer",
                running=True,
            )
    if service is None:
        print(
            "No service could be found. The pipeline will be run first to create a service."
        )
        main()
    return service

# MODEL_FILE = os.getenv('MODEL_FILE', 'lin_reg.bin')
# the address where the data will be sent for monitoring using evidently 
EVIDENTLY_SERVICE_ADDRESS = os.getenv('EVIDENTLY_SERVICE', 'http://127.0.0.1:5000')
# MANGODB for storing the data before it goes to the premotheous and evidently or monitoring service
MONGODB_ADDRESS = os.getenv("MONGODB_ADDRESS", "mongodb://127.0.0.1:27017")

# creating a flask
app = Flask('duration')
# creating the database
mongo_client = MongoClient(MONGODB_ADDRESS)
# getting the database and calling the predictive service
db = mongo_client.get_database("prediction_service")
# getting the data that will be stored and sent to evidently
collection = db.get_collection("data")

model_service = get_model_as_service()

@app.route('/predict', methods=['POST'])
def predict():
    # getting the json file that has the records
    record = request.get_json()
    # predict using the model
    y_pred = model_service.predict(record)
    # results stored in a form of a dictionary
    result = {
        'Price': float(y_pred),
    }
    # create a function that will save the data predicted to the database
    save_to_db(record, float(y_pred))
    # a function to send the data to evidently service
    send_to_evidently_service(record, float(y_pred))
    # return tthe results in A jsonify file
    return jsonify(result)

# function that takse in the records and teh predicted results, store them in db
def save_to_db(record, prediction):
    # create a copy of the record
    rec = record.copy()
    #in the records dataframe,create a column  for the predicted values
    rec['prediction'] = prediction
    # add the recods to database
    collection.insert_one(rec)

# function to get the records and the predicted records to the monitoring service
def send_to_evidently_service(record, prediction):
    # make a copy of the records
    rec = record.copy()
    # create a column of the predicted values
    rec['prediction'] = prediction
    # qwe neeed ri post the records in a json file to the evidently service address
    requests.post(f"{EVIDENTLY_SERVICE_ADDRESS}/iterate/laptop", json=[rec])


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)