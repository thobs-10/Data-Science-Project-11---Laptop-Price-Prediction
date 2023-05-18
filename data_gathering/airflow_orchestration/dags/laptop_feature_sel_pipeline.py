# import necessary stuff for airflow to run
from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
from airflow.operators.mysql_operator import MySqlOperator
from airflow.operators.email_operator import EmailOperator


# import steps for the DAG
import sys
import os
#sys.path.append(r"C:\Users\Cash Crusaders\Desktop\My Portfolio\Projects\Data Science Projects\Data Science Project 11 - Laptop Price Prediction\data_gathering")
sys.path.append("C:/Users/Cash Crusaders/Desktop/My Portfolio/Projects/Data Science Projects/Data Science Project 11 - Laptop Price Prediction/data_gathering")
# feature selection steps
from feature_selection import seperate_dataset
from feature_selection import label_encode
from feature_selection import feature_importance
from feature_selection import drop_top_correlated_features
from feature_selection import drop_correlated
from feature_selection import split_for_PCA
from feature_selection import principal_component_analysis
from feature_selection import get_most_important_features
from feature_selection import feature_scaling



default_args = {"owner": "thobela", "start_date": datetime(2023, 3, 17, 5)}

# DAG for feature feature selection
with DAG(
    dag_id="laptop-feature-engineering",
    default_args=default_args,
    description='This is a feature engineering dag',
    schedule_interval='@daily',
    catchup=True) as dag:
    # nodes\steps
    seperate_dataset = PythonOperator(
        task_id="seperate_data",
        python_callable=seperate_dataset,
        retries=2,
        retry_delay=timedelta(seconds=15)
    )
    label_encode = PythonOperator(
        task_id="label_encode",
        python_callable=label_encode
        
    )
    feature_importance = PythonOperator(
        task_id="feature_importance",
        python_callable=feature_importance
        
    )
    drop_top_correlated_features = PythonOperator(
        task_id="drop_top_correlated_features",
        python_callable=drop_top_correlated_features 
    )
    drop_correlated = PythonOperator(
        task_id="drop_correlated",
        python_callable=drop_correlated 
    )
    split_for_PCA = PythonOperator(
        task_id="split_for_PCA",
        python_callable=split_for_PCA 
    )
    principal_component_analysis = PythonOperator(
        task_id="pca",
        python_callable=principal_component_analysis 
    )
    get_most_important_features = PythonOperator(
        task_id="get_most_important_features",
        python_callable=get_most_important_features 
    )
    feature_scaling = PythonOperator(
        task_id="feature_scaling",
        python_callable=feature_scaling 
    )

    seperate_dataset >> label_encode