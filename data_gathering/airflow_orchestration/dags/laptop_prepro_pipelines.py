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
#from data_gathering.get_data import ingest_data
from get_data import ingest_data
from preprocessing import fix_column_datatypes
from preprocessing import create_column
from preprocessing import preprocess_features
from preprocessing import drop_unwanted_columns
from preprocessing import apply_fetch_processors


default_args = {"owner": "thobela", "start_date": datetime(2023, 3, 17, 5)}

with DAG(
    dag_id="laptop-preprocessing",
    default_args=default_args,
    description='This is a preprocessing dag',
    schedule_interval='@daily',
    catchup=True) as dag:
   # begining of nodes
    check_file = BashOperator(
        task_id="check_file",
        bash_command="shasum C:\\Users\\Cash Crusaders\\Desktop\\My Portfolio\\Projects\\Data Science Projects\\Data Science Project 11 - Laptop Price Prediction\\dataset\\laptop_data.csv",
        retries=2,
        retry_delay=timedelta(seconds=15)
        )
    
    ingest_data = PythonOperator(
        task_id="get_file",
        python_callable=ingest_data,
        retries=2,
        retry_delay=timedelta(seconds=15)
        )

    fix_column_datatypes = PythonOperator(
        task_id = "fix_columns",
        python_callable = fix_column_datatypes
        )
    create_column = PythonOperator(
        task_id = "create_columns",
        python_callable = create_column
        )
    preprocess_features = PythonOperator(
        task_id = "process_features",
        python_callable = preprocess_features
        )
    drop_unwanted_columns =  PythonOperator(
        task_id = "drop_features",
        python_callable = drop_unwanted_columns
        )
    check_file >> ingest_data >> fix_column_datatypes >> create_column >> preprocess_features >> drop_unwanted_columns

