import numpy as np
import pandas as pd
from numpy import savetxt
from numpy import loadtxt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from zenml.steps import Output, step

from zenml.integrations.airflow.flavors.airflow_orchestrator_flavor import AirflowOrchestratorSettings

from zenml.config import DockerSettings
from zenml.integrations.constants import AIRFLOW
def read_data():
    df = pd.read_parquet('C:\\Users\\Cash Crusaders\\Desktop\\My Portfolio\\Projects\Data Science Projects\\Data Science Project 11 - Laptop Price Prediction\\dataset\\preprocessed_data\\preprocessed.parquet.gzip')
    return df

    # Using the operator for a single step
@step
def memory_engineering()->Output(
    df=pd.DataFrame
):
    df = read_data()
    #df = ti.xcom(task_ids = '')
    df['Memory'] = df['Memory'].astype(str).replace('\.0', '', regex=True)
    # replace the GB with nothing and TB wth '000'
    df["Memory"] = df["Memory"].str.replace('GB', '')
    df["Memory"] = df["Memory"].str.replace('TB', '000')
    #split the values of the column using the '+' sign and taking the values to the a list
    new = df["Memory"].str.split("+", n = 1, expand = True)
    # get the values from the list of values in column  Memory
    df["first"]= new[0]
    df["first"]=df["first"].str.strip() # strip the list property from the values in the row
    # get the second valuee of the list
    df["second"]= new[1]
    # #get the value from the column first, for each value in row is the value is HDD place 1 in its place if not 0,
    return df
@step
def SDD_engineering(df:pd.DataFrame)->Output(
    df=pd.DataFrame
):
    # df = ti.xcom(task_ids = '')
    # same applies to  SDD
    df["Layer1HDD"] = df["first"].apply(lambda x: 1 if "HDD" in x else 0)
    df["Layer1SSD"] = df["first"].apply(lambda x: 1 if "SSD" in x else 0)
    #same applied to hybrid and flash storage
    df["Layer1Hybrid"] = df["first"].apply(lambda x: 1 if "Hybrid" in x else 0)
    df["Layer1Flash_Storage"] = df["first"].apply(lambda x: 1 if "Flash Storage" in x else 0)
    # replace D with nothing
    df['first'] = df['first'].str.replace(r'\D', '')
    # fill iin  the missing values with 0
    df["second"].fillna("0", inplace = True)

    df["Layer2HDD"] = df["second"].apply(lambda x: 1 if "HDD" in x else 0)
    df["Layer2SSD"] = df["second"].apply(lambda x: 1 if "SSD" in x else 0)
    df["Layer2Hybrid"] = df["second"].apply(lambda x: 1 if "Hybrid" in x else 0)
    df["Layer2Flash_Storage"] = df["second"].apply(lambda x: 1 if "Flash Storage" in x else 0)

    df['second'] = df['second'].str.replace(r'\D', '')

    df["first"] = df["first"].astype(int)
    df["second"] = df["second"].astype(int)

    return df
@step
def HDD_engineering(df:pd.DataFrame)->Output(
    df=pd.DataFrame
):
    # df = ti.xcom(task_ids = '')
    df["HDD"]=(df["first"]*df["Layer1HDD"]+df["second"]*df["Layer2HDD"])
    df["SSD"]=(df["first"]*df["Layer1SSD"]+df["second"]*df["Layer2SSD"])
    df["Hybrid"]=(df["first"]*df["Layer1Hybrid"]+df["second"]*df["Layer2Hybrid"])
    df["Flash_Storage"]=(df["first"]*df["Layer1Flash_Storage"]+df["second"]*df["Layer2Flash_Storage"])

    return df
@step
def drop_columns(df:pd.DataFrame)->Output(
    df=pd.DataFrame
):
    # df = ti.xcom(task_ids = '')
    df.drop(columns=['first', 'second', 'Layer1HDD', 'Layer1SSD', 'Layer1Hybrid',
       'Layer1Flash_Storage', 'Layer2HDD', 'Layer2SSD', 'Layer2Hybrid',
       'Layer2Flash_Storage'],inplace=True)
    
    df.drop(columns=['Memory'],inplace=True)

    #bremove the negative correlation columns
    df.drop(columns=['Hybrid','Flash_Storage'],inplace=True)
    return df
@step
def GPU_engineering(df:pd.DataFrame)->Output(
    df=pd.DataFrame
):
    # df = ti.xcom(task_ids = '')
    df['Gpu brand'] = df['Gpu'].apply(lambda x:x.split()[0])
    # remove the value of gpu brand with ARM
    df = df[df['Gpu brand'] != 'ARM']
    df.drop(columns=['Gpu'],inplace=True)
    return df

def cat_os(inp):
    if inp == 'Windows 10' or inp == 'Windows 7' or inp == 'Windows 10 S':
        return 'Windows'
    elif inp == 'macOS' or inp == 'Mac OS X':
        return 'Mac'
    else:
        return 'Others/No OS/Linux'
    
@step 
def apply_cat_os(df:pd.DataFrame)->Output(
    df=pd.DataFrame
):
    # df = ti.xcom(task_ids = '')
    df['os'] = df['OpSys'].apply(cat_os)
    df.drop(columns=['OpSys'],inplace=True)
    #df.to_parquet('C:\\Users\\Cash Crusaders\\Desktop\\My Portfolio\\Projects\\Data Science Projects\\Data Science Project 11 - Laptop Price Prediction\\dataset\\feature_engineered_data\\feature_engineeredV2.parquet.gzip',compression='gzip')
    return df

@step
def seperate_dataset(df:pd.DataFrame)->Output(
    X = pd.DataFrame,
    y = pd.Series
):
    # df = ti.xcom(task_ids = '')
    X = df.drop(columns=['Price'])
    y = np.log(df['Price'])
    return X,y

# @step
# def label_encode(X:pd.DataFrame)->Output(
#    X=np.ndarray
# ):
#     ct = ColumnTransformer(transformers=[('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])],remainder='passthrough')
#     X = ct.fit_transform(X)
#     return X

@step
def split_dataset(X:pd.DataFrame,y:pd.Series)->Output(
    X_train= pd.DataFrame,
    X_test = pd.DataFrame,
    y_train = pd.Series,
    y_test = pd.Series
):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
    return X_train,X_test, y_train, y_test

@step
def save_dataset( X_train:pd.DataFrame,X_test:pd.DataFrame, y_train:pd.Series, y_test:pd.Series)->Output(
    
):
    # change the values to be dataframes
    
    # X_train_df=pd.DataFrame(X_train)
    # X_test_df = pd.DataFrame(X_test)
    # y_train_df =pd.DataFrame(y_train)
    # y_test_df =pd.DataFrame(y_test)

    #save to csv
    #savetxt('C:\\Users\\Cash Crusaders\\Desktop\\My Portfolio\\Projects\\Data Science Projects\\Data Science Project 11 - Laptop Price Prediction\\dataset\\feature_engineered_data\\X_train.csv',X_train )

    X_train.to_parquet('C:\\Users\\Cash Crusaders\\Desktop\\My Portfolio\\Projects\\Data Science Projects\\Data Science Project 11 - Laptop Price Prediction\\dataset\\feature_engineered_data\\X_train_df.parquet.gzip')
    X_test.to_parquet('C:\\Users\\Cash Crusaders\\Desktop\\My Portfolio\\Projects\\Data Science Projects\\Data Science Project 11 - Laptop Price Prediction\\dataset\\feature_engineered_data\\X_test_df.parquet')
    y_train.to_csv('C:\\Users\\Cash Crusaders\\Desktop\\My Portfolio\\Projects\\Data Science Projects\\Data Science Project 11 - Laptop Price Prediction\\dataset\\feature_engineered_data\\y_train_df.csv')
    y_test.to_csv('C:\\Users\\Cash Crusaders\\Desktop\\My Portfolio\\Projects\\Data Science Projects\\Data Science Project 11 - Laptop Price Prediction\\dataset\\feature_engineered_data\\y_test_df.csv')


