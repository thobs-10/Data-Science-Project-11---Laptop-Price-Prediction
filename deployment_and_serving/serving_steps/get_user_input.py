
import numpy as np
import pandas as pd
import math
from zenml.steps import step, Output

@step
def get_input(company,laptop_type,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os,resolution,screen_size):
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size
    query = np.array([company,laptop_type,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])
    query.reshape(1,12)
    return query
