# import necessary stuff for airflow to run
from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
from airflow.operators.mysql_operator import MySqlOperator
from airflow.operators.email_operator import EmailOperator
from datetime import datetime,timedelta

# import steps for the DAG
import sys
import os
sys.path.append("C:/Users/Cash Crusaders/Desktop/My Portfolio/Projects/Data Science Projects/Data Science Project 11 - Laptop Price Prediction/data_gathering")
# feature engineering steps
from feature_engineering import memory_engineering
from feature_engineering import SDD_engineering
from feature_engineering import HDD_engineering
from feature_engineering import drop_columns
from feature_engineering import GPU_engineering
from feature_engineering import apply_cat_os
from feature_engineering import seperate_dataset


default_args = {"owner": "thobela", "start_date": datetime(2023, 3, 17, 5)}
# DAG for feature transformation
with DAG(
    dag_id="laptop-feature-engineering",
    default_args=default_args,
    description='This is a feature engineering dag',
    schedule_interval='@daily',
    catchup=True) as dag:
    # nodes\steps
    memory_engineering = PythonOperator(
        task_id="memory_feature",
        python_callable=memory_engineering,
        retries=2,
        retry_delay=timedelta(seconds=15)
        )
    SDD_engineering =  PythonOperator(
        task_id = "ssd_feature",
        python_callable = SDD_engineering
        )
    HDD_engineering =  PythonOperator(
        task_id = "hdd_feature",
        python_callable = HDD_engineering
        )
    drop_columns =  PythonOperator(
        task_id = "drop_features_engineering",
        python_callable = drop_columns
        )
    GPU_engineering =  PythonOperator(
        task_id = "gpu_feature",
        python_callable = GPU_engineering
        )
    apply_cat_os =  PythonOperator(
        task_id = "drop_features",
        python_callable = apply_cat_os
        )
    # seperate_dataset = PythonOperator(
    #     task_id = "separate_dataset",
    #     python_callable = seperate_dataset
    #     )
    
    memory_engineering >> SDD_engineering >> HDD_engineering >> drop_columns >> GPU_engineering >> apply_cat_os 
