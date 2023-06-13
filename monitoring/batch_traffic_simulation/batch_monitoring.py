import json
import os
import pickle

import pandas as pd
import numpy as np
from prefect import flow, task
from pymongo import MongoClient
import pyarrow.parquet as pq
from datetime import datetime, timedelta

from sklearn.feature_extraction import DictVectorizer

from evidently import ColumnMapping

from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab,RegressionPerformanceTab

from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection, RegressionPerformanceProfileSection
# custom built report
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset, RegressionPreset
from evidently.test_suite import TestSuite
from evidently.tests import *
from evidently.test_preset import NoTargetPerformanceTestPreset
from evidently.test_preset import DataQualityTestPreset
from evidently.test_preset import DataStabilityTestPreset
from evidently.test_preset import DataDriftTestPreset
from evidently.test_preset import RegressionTestPreset




def upload_target(collection_db_filename, filename):
    # client = MongoClient("mongodb://localhost:27018/")
    # collection = client.get_database("prediction_service").get_collection("data")
    df = pd.read_parquet(collection_db_filename)
    df_target = pd.read_csv(filename)
    # df_target = df_target.transpose()
    # df_target = df_target.iloc[1:]
    df['target'] = df_target
    df.to_parquet("C:\\Users\\Cash Crusaders\\Desktop\\My Portfolio\\Projects\\Data Science Projects\\Data Science Project 11 - Laptop Price Prediction\\monitoring\\batch_traffic_simulation\\current_data.parquet")
    # with open(filename) as f_target:
    #     for line in f_target.readlines():
    #         row = line.split(",")
    #         df['target'] = row[1]
            # collection.update_one({"id": row[0]}, {"$set": {"target": float(row[1])}})
    # client.close()

def load_reference_data(filename, target_filename):
    MODEL_FILE = os.getenv('MODEL_FILE', 'rf_reg.pkl')
    with open(MODEL_FILE, 'rb') as f_in:
        model = pickle.load(f_in)

    reference_data = pd.read_parquet(filename)
    y_test = pd.read_csv(target_filename)
    # Create features
    reference_data.drop(columns=['Unnamed: 0'],inplace=True)
    reference_data.drop(columns=['Cpu'],inplace=True)
    reference_data.drop(columns=['Gpu'],inplace=True)
    reference_data.drop(columns=['Cpu Name'],inplace=True)
    y_test.drop(columns=['Unnamed: 0'],inplace=True)
    # transformation
    TypeName_map=reference_data['TypeName'].value_counts().to_dict()
    reference_data['TypeName']=reference_data['TypeName'].map(TypeName_map)
    # company
    company_map=reference_data['Company'].value_counts().to_dict()
    reference_data['Company']=reference_data['Company'].map(company_map)
    #Cpu brand
    cpu_brand_map=reference_data['Cpu brand'].value_counts().to_dict()
    reference_data['Cpu brand']=reference_data['Cpu brand'].map(cpu_brand_map)
    # Gpu brand
    gpu_brand_map=reference_data['Gpu brand'].value_counts().to_dict()
    reference_data['Gpu brand']=reference_data['Gpu brand'].map(gpu_brand_map)
    # os
    os_map=reference_data['os'].value_counts().to_dict()
    reference_data['os']=reference_data['os'].map(os_map)
    # add target column
    reference_data['target'] = y_test
    features = ['Company', 'TypeName', 'Ram', 'Weight','Touchscreen','Ips','ppi','Cpu brand','HDD','SSD','Gpu brand','os']
    x_pred = reference_data[features]
    x_pred_numpy = np.array(x_pred)
    reference_data['prediction'] = model.predict(x_pred_numpy)
    return reference_data

def fetch_data(filename):
    # client = MongoClient("mongodb://localhost:27018/")
    # data = client.get_database("prediction_service").get_collection("data").find()
    # df = pd.DataFrame(list(data))
    df = pd.read_parquet(filename)
    df.rename(columns={'predictions':'target'}, inplace=True)
    return df

def run_evidently(ref_data, cur_data):
    column_mapping = ColumnMapping()
    column_mapping.target = 'target'
    #column_mapping.categorical_features = []
    column_mapping.numerical_features = [ 'Company', 'TypeName','Cpu brand','Gpu brand','os','Ram', 'Weight','Touchscreen','Ips','ppi','HDD','SSD']
    
    drift_report = Report(metrics=[DataDriftPreset(), TargetDriftPreset(), DataQualityPreset(), RegressionPreset()])
    drift_report.run(reference_data=ref_data, current_data=cur_data, column_mapping=column_mapping)
    # drift_report
    test_report = TestSuite(tests=[RegressionTestPreset(), DataStabilityTestPreset(), DataQualityTestPreset()])
    test_report.run(reference_data=ref_data, current_data=cur_data)
    return drift_report, test_report


def save_report(result):
    client = MongoClient("mongodb://localhost:27018/")
    client.get_database("prediction_service").get_collection("report").insert_one(result[0])

def save_html_report(drift_report, test_report):
    cur_date = datetime.now()
    
    drift_report.save_html("evidently_report_01.html")
    test_report.save_html(f"evidently_test_report_01.html")
    # report.save("evidently_report_example.html")

def batch_analyze():
    upload_target("collection_db.parquet","predicted_target.csv")
    ref_data = load_reference_data("X_test_df.parquet", "y_test_df.csv")
    cur_data = fetch_data("current_data.parquet")
    report_result, test_report = run_evidently(ref_data, cur_data)
    # save_report(result)
    save_html_report(report_result, test_report)
    print("Successfully saved...")

batch_analyze()