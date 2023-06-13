import json
import uuid
from datetime import datetime
from time import sleep
import pandas as pd
import numpy as np

import pyarrow.parquet as pq
import requests

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# table = pq.read_table("X_test_df.parquet")
# data = table.to_pylist()
data = pd.read_csv("y_test_df.csv")
pred_data = pd.read_parquet("X_test_df.parquet")


pred_data.drop(columns=['Unnamed: 0'],inplace=True)
pred_data.drop(columns=['Cpu'],inplace=True)
pred_data.drop(columns=['Gpu'],inplace=True)
pred_data.drop(columns=['Cpu Name'],inplace=True)
data.drop(columns=['Unnamed: 0'],inplace=True)

# x_train_data.drop(columns=['Unnamed: 0'],inplace=True)
# x_train_data.drop(columns=['Cpu'],inplace=True)
# x_train_data.drop(columns=['Gpu'],inplace=True)
# x_train_data.drop(columns=['Cpu Name'],inplace=True)

# X- train
TypeName_map=pred_data['TypeName'].value_counts().to_dict()
pred_data['TypeName']=pred_data['TypeName'].map(TypeName_map)
# company
company_map=pred_data['Company'].value_counts().to_dict()
pred_data['Company']=pred_data['Company'].map(company_map)
#Cpu brand
cpu_brand_map=pred_data['Cpu brand'].value_counts().to_dict()
pred_data['Cpu brand']=pred_data['Cpu brand'].map(cpu_brand_map)
# Gpu brand
gpu_brand_map=pred_data['Gpu brand'].value_counts().to_dict()
pred_data['Gpu brand']=pred_data['Gpu brand'].map(gpu_brand_map)
# os
os_map=pred_data['os'].value_counts().to_dict()
pred_data['os']=pred_data['os'].map(os_map)

# X_test



# data = dataframe.values.tolist()
# class DateTimeEncoder(json.JSONEncoder):
#     def default(self, o):
#         if isinstance(o, datetime):
#             return o.isoformat()
#         return json.JSONEncoder.default(self, o)

# read the target csv file
with open("predicted_target.csv", 'w') as f_target:
    # go through each row
    f_target.write(f"price\n")
    for index, row in data.iterrows():
        # create a unique id
        # row['id'] = str(uuid.uuid4())
        price = row['series']
        # unique_id = row["id"]
        f_target.write(f"{price}\n")

# column_transformer = ColumnTransformer(transformers=[
#             ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
#         ],remainder='passthrough')
    
# X_train_tranformed = np.array(column_transformer.fit_transform(x_train_data))
# input_data_tranformed = np.array(column_transformer.transform(pred_data))



input_data_tranformed = np.array(pred_data)
serialized_input_data = json.dumps({'data':input_data_tranformed.tolist()})
resp = requests.post("http://127.0.0.1:9696/predict",
                            headers={"Content-Type": "application/json"},
                            data=serialized_input_data)
# print the price value from the response of the request from the endpoint
my_dict = json.loads(resp.content)
print(my_dict[0])
# need to convert this into a df and save it for batch monitoring.
df_dict = pd.DataFrame.from_dict(my_dict)
df_dict.to_parquet('C:\\Users\\Cash Crusaders\\Desktop\\My Portfolio\\Projects\\Data Science Projects\\Data Science Project 11 - Laptop Price Prediction\\monitoring\\online_monitoring\\collection_db.parquet')
# pred_data['Pred_Price'] = my_dict['predicted_price']
# for index, row in pred_data.iterrows():

#     row_df = pd.DataFrame([row])
#     x_train_row_df = row_df.copy()

#     column_transformer = ColumnTransformer(transformers=[
#             ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
#         ],remainder='passthrough')
    
#     # X_train_tranformed = np.array(column_transformer.fit_transform(x_train_row_df))
#     # input_data_tranformed = np.array(column_transformer.transform(row_df))

#     input_data_tranformed = np.array(row_df)
#     serialized_input_data = json.dumps({'data':input_data_tranformed.tolist()})
#     resp = requests.post("http://127.0.0.1:9696/predict",
#                              headers={"Content-Type": "application/json"},
#                              data=serialized_input_data)
#     # print the price value from the response of the request from the endpoint
#     print(f"prediction: {resp.text}")
#     # sleep for 1 second and run again for another row
#     sleep(1)
