import numpy as np
import pandas as pd



from zenml.steps import Output, step

from zenml.integrations.airflow.flavors.airflow_orchestrator_flavor import AirflowOrchestratorSettings

from zenml.config import DockerSettings
from zenml.integrations.constants import AIRFLOW

docker_settings = DockerSettings(
    required_integrations = [AIRFLOW],
    requirements=["apache-airflow-providers-docker"]
)

airflow_settings = AirflowOrchestratorSettings(
    operator="docker",  # or "kubernetes_pod"
    # Dictionary of arguments to pass to the operator __init__ method
    operator_args={}
)


# Using the operator for a single step
@step(enable_cache=True, enable_artifact_metadata=True,settings={"orchestrator.airflow": airflow_settings,"docker": docker_settings})

def fix_column_datatypes(df:pd.DataFrame)->Output(
        df=pd.DataFrame
):

    #df = ti.xcom_pull(task_ids='get_file')
    # we want to remove the GB in values of the ram column
    df['Ram'] = df['Ram'].str.replace('GB','')
    # do the same thing for weigght
    df['Weight'] = df['Weight'].str.replace('kg','')

    # change the data types of Ram and weight to be in and float
    df['Ram'] = df['Ram'].astype('int32')
    df['Weight'] = df['Weight'].astype('float32')
    return df
# Using the operator for a single step
@step(enable_cache=True, enable_artifact_metadata=True,settings={"orchestrator.airflow": airflow_settings,"docker": docker_settings})
def create_column(df:pd.DataFrame)->Output(
        df=pd.DataFrame
):

    #df = ti.xcom_pull(task_ids='fix_columns')
    # creating a column for touch screen devices, if the device is a touch screen then 1, if not then 0
    df['Touchscreen'] = df['ScreenResolution'].apply(lambda x:1 if 'Touchscreen' in x else 0)

    # creating a column for Ips devices, if the device is a Ips then 1, if not then 0
    df['Ips'] = df['ScreenResolution'].apply(lambda x:1 if 'IPS' in x else 0)

    # this will first take the entries and split the using the x value, after take the two values of the list and make new dataframe
    new_df = df['ScreenResolution'].str.split('x', n=1, expand=True)

    df['x_res'] = new_df[0]
    df['y_res'] = new_df[1]

    return df
# Using the operator for a single step
@step(enable_cache=True, enable_artifact_metadata=True,settings={"orchestrator.airflow": airflow_settings,"docker": docker_settings})
def preprocess_features(df:pd.DataFrame)->Output(
        df=pd.DataFrame
):

    #df = ti.xcom_pull(task_ids='create_columns')
    # first replace commas with nothing for every value in x_res column, after find the numbers in the values of the column 
    # place them in a list, then apply lambda for all the vallues in that column to get the single value for x_res
    df['x_res'] = df['x_res'].str.replace(',','').str.findall(r'(\d+\.?\d+)').apply(lambda x:x[0])

    df['x_res'] = df['x_res'].astype('int')
    df['y_res'] = df['y_res'].astype('int')

    # We not using x_res and y_res because there is multicolinearity between them
    df['ppi'] = (((df['x_res']**2) + (df['y_res']**2))**0.5/df['Inches']).astype('float')

    return df
# Using the operator for a single step
@step(enable_cache=True, enable_artifact_metadata=True,settings={"orchestrator.airflow": airflow_settings,"docker": docker_settings})
def drop_unwanted_columns(df:pd.DataFrame)->Output(
        df=pd.DataFrame
):
    #df = ti.xcom_pull(task_ids='process_features')
    # since the screen resolution has been decoded into different columns or features we can remove it now
    df.drop(columns=['ScreenResolution'],inplace=True)

    # drop the x_res and y_res since we used them to calculate the ppi and they will no longer be necessary
    df.drop(columns=['Inches','x_res','y_res'], axis=1, inplace=True)

    df['Cpu Name'] = df['Cpu'].apply(lambda x: " ".join(x.split()[0:3]))

    return df

def fetch_processors(text):
    if text == 'Intel Core i7' or text == 'Intel Core i5' or text == 'Intel Core i3':
        return text
    elif text.split()[0] == 'Intel':
        return 'Other Intel Processor'
    else:
        return 'AMD Processor'
    
# Using the operator for a single step
@step(name="last_prep_task",enable_cache=True, enable_artifact_metadata=True,settings={"orchestrator.airflow": airflow_settings,"docker": docker_settings})
def apply_fetch_processors(df:pd.DataFrame)->Output(
        df=pd.DataFrame,
):
    #df = ti.xcom_pull(task_ids='drop_features')
    df['Cpu brand'] = df['Cpu Name'].apply(fetch_processors)
    # save to the dataset as paquet file
    df.to_parquet('C:\\Users\\Cash Crusaders\\Desktop\\My Portfolio\\Projects\\Data Science Projects\\Data Science Project 11 - Laptop Price Prediction\\dataset\\preprocessed_data\\preprocessed.parquet.gzip',compression='gzip')
    return df

