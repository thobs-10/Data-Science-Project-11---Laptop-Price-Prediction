# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 17:05:20 2021

@author: viswa
"""

from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
from airflow.operators.mysql_operator import MySqlOperator
from airflow.operators.email_operator import EmailOperator
from datetime import datetime,timedelta
from pre_process import process_data
from pre_processing import pre_process



default_args = {"owner": "airflow", "start_date": datetime(2021, 6, 15)}

with DAG(dag_id="analytics",default_args=default_args,schedule_interval='@daily',catchup=True) as dag:
   
    check_file = BashOperator(
        task_id="check_file",
        bash_command="shasum ~/ip_files/or.csv",
        retries=2,
        retry_delay=timedelta(seconds=15)
        )
    
    pre_process = PythonOperator(
        task_id = "pre",
        python_callable = pre_process
        )
    
    groupbys = PythonOperator(
        task_id="aggre",
        python_callable=process_data
        )
    
    create_table = MySqlOperator(
        task_id='create_table', 
        mysql_conn_id="mysql_db1", 
        sql="CREATE table IF NOT EXISTS aggre_res (stock_code varchar(100) NULL,descb varchar(100) NULL,country varchar(100) NULL,total_price varchar(100) NULL)"
        )

    
    insert = MySqlOperator(
        task_id='insert_db', 
        mysql_conn_id="mysql_db1", 
        sql="LOAD DATA  INFILE '/var/lib/mysql-files/fin.csv' INTO TABLE aggre_res FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n' IGNORE 1 ROWS;")
    
    email = EmailOperator(task_id='send_email',
        to='viswatejaster@gmail.com',
        subject='Daily report generated',
        html_content=""" <h1>Congratulations! Your store reports are ready.</h1> """,
        )

    check_file >> pre_process >> groupbys >> create_table >> insert >> email