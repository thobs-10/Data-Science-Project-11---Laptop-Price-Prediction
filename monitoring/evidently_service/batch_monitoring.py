import json
import pickle

import pandas as pd
from pymongo import MongoClient
import pyarrow.parquet as pq

from evidently import ColumnMapping
from evidently.report import Report
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab,RegressionPerformanceTab

from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection, RegressionPerformanceProfileSection

from deployment_and_serving.run_deployment_serving_pipeline import main
from zenml.services import load_last_service_from_step

features = ["company","laptop_type","ram","weight","touchscreen","ips","ppi","cpu","hdd","ssd","gpu","os"]
numerical_features = ["ram","weight","touchscreen","ips","ppi","hdd","ssd"]
categorical_features = ["company","laptop_type","cpu","gpu","os"]
datetime_features = None


def upload_target(filename):
    client = MongoClient("mongodb://localhost:27018/")
    collection = client.get_database("prediction_service").get_collection("data")
    with open(filename) as f_target:
        for line in f_target.readlines():
            row = line.split(",")
            collection.update_one({"id": row[0]}, {"$set": {"target": float(row[1])}})
    client.close()

def load_reference_data(reference_baseline_filename):

    model_service = load_last_service_from_step(
        pipeline_name="continuous_deployment_pipeline",
        step_name="model_deployer",
        running=True,
    )
     
    if model_service is None:
        print(
            "No service could be found. The pipeline will be run first to create a service."
        )
        main()

    reference_data = pd.read_csv(reference_baseline_filename)
    # Create features
    x_pred = reference_data[features]
    reference_data['prediction'] = model_service.predict(x_pred)
    return reference_data


def fetch_data():
    client = MongoClient("mongodb://localhost:27018/")
    data = client.get_database("prediction_service").get_collection("data").find()
    df = pd.DataFrame(list(data))
    return df



def run_evidently(ref_data, data):
    
    profile = Profile(sections=[DataDriftProfileSection(), RegressionPerformanceProfileSection()])

    mapping = ColumnMapping(prediction="prediction", numerical_features=numerical_features,
                            categorical_features=categorical_features,
                            datetime_features=datetime_features)
    
    profile.calculate(ref_data, data, mapping)

    dashboard = Dashboard(tabs=[DataDriftTab(), RegressionPerformanceTab(verbose_level=0)])
    dashboard.calculate(ref_data, data, mapping)
    return json.loads(profile.json()), dashboard

def save_report(result):
    client = MongoClient("mongodb://localhost:27018/")
    client.get_database("prediction_service").get_collection("report").insert_one(result[0])


def save_html_report(result):
    result[1].save("laptop_price prediction_evidently_report.html")

def batch_analyze():
    upload_target("predicted_target.csv")
    ref_baseline_data = load_reference_data("dataset\\feature_engineered_data\\y_test_df.csv")
    current_pred_data = fetch_data()
    result = run_evidently(ref_baseline_data, current_pred_data)
    save_report(result)
    save_html_report(result)

# batch_analyze()

if __name__ == "__main__":
    batch_analyze()

