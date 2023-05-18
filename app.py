import streamlit as st
import numpy as np
import pickle
import math
from typing import cast
import mlflow
import requests
import json
from typing import Tuple
import logging


# read the saved pickle packages
# pipeline = pickle.load(open('dataset/pipe.pkl','rb'))
df = pickle.load(open('dataset/df.pkl','rb'))
service_url = "http://localhost:3000/predict"
# def streamlit_app():

st.title("Laptop Prediction")

#laptop brand select box
company = st.selectbox('Brand', df['Company'].unique())
#type of laptop
laptop_type = st.selectbox('Type', df['TypeName'].unique())
#RAM
ram = st.selectbox('RAM(in GB)', [2,4,6,8,12,16,24,32,64])
#weight
weight = st.number_input('Weight of laptop')
# touchscreen
touchscreen = st.selectbox('Touchscreen',['No','Yes'])
# IPS
ips = st.selectbox('IPS',['No','Yes'])
#screen size
screen_size = st.number_input('Screen Size')
#Resolution
resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900',
                                            '3840x1800','2888x1800','2560x1600','2560x1440','2384x1440'])
#CPU
cpu = st.selectbox('CPU', df['Cpu brand'].unique())
#hardrive
hdd = st.selectbox('HDD(in GB)', [0,128,256,512,1024,2048])
# SSD
ssd = st.selectbox('SSD(in GB)', [0,8,128,256,512,1024])
#GPU
gpu = st.selectbox('GPU', df['Gpu brand'].unique())
#operating system(os)
os =  st.selectbox('OS', df['os'].unique())

if st.button('Predict Price'):
    #query point
    ppi = None
    if touchscreen=='Yes':
        touchscreen= 1
    else:
        touchscreen = 0
    
    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    # calculate ppi
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size
    query = np.array([company,laptop_type,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])

    query.reshape(1,12)

    serialized_input_data = json.dumps(query.tolist())
    response = requests.post(
        service_url,
        data=serialized_input_data,
        headers={"content-type":"application/json"}
    )
    pred = response.text
    # service = load_last_service_from_step(
    #     pipeline_name="continuous_deployment_pipeline",
    #     step_name="model_deployer",
    #     running=True,
    # )
    # if service is None:
    #     st.write(
    #         "No service could be found. The pipeline will be run first to create a service."
    #     )
    #     main()

    # pred = service.predict(query)
    # st.success(
    #     "Your Customer Satisfactory rate(range between 0 - 5) with given product details is :-{}".format(
    #         pred
    #     )
    # )
    st.title("The predicted price of this configuration is " + str(pred[0]))
    # st.title("The predicted price of this configuration is " + str(int(np.exp(pred[0]))))
    # st.title("The predicted price of this configuration is " + str(int(np.exp(pipeline.predict(query)[0]))))

# if __name__=="__main__":
#     streamlit_app()

