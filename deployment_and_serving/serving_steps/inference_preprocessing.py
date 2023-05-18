import numpy as np
import pandas as pd
from sklearn import StandardScaler
from sklearn.preprocessing import StandardScaler
from zenml.steps import step, Output

@step
def preprocess_input(data):
    """preprocess the incoming user input"""
    # scale the features
    sc = StandardScaler()
    # transformed_data = sc.transform(data)
    return data