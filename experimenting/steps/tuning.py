import argparse
import os
import pickle
import pandas as pd

import mlflow
import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval
from hyperopt.pyll import scope
from sklearn.metrics import mean_squared_error, r2_score
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from zenml.steps import step, Output, step_output
from zenml.client import Client
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.experiment_trackers import (
    MLFlowExperimentTracker,
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor,ExtraTreesRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

mlflow.set_tracking_uri("sqlite:///laptop.db")
# mlflow.set_experiment("random-forest-hyperopt")

# get the stack experiment tracking uri
#MLFLOW_TRACKING_URI = get_tracking_uri()
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
# MLFLOW_TRACKING_URI = "./mlruns"
# assign a name for this hyperparameter tuning experiment
TRAINING_EXPERIMENT_NAME = "training-laptop-experiment-1"
EXPERIMENT_NAME = "hyper-optimization-experiment-1"
# set the stack experiment tracking uri to be the current working one
# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
# set the current experiment name to be the current hyperparameter tuning one
mlflow.set_experiment(EXPERIMENT_NAME)

#get the experiment tracker of the active stack from the stack components
experiment_tracker = Client().active_stack.experiment_tracker

# cjecking if the current experiment tracker of the active stack os present
if not experiment_tracker or not isinstance(
    experiment_tracker, MLFlowExperimentTracker
):
    raise RuntimeError(
        "Your active stack needs to contain a MLFlow experiment tracker for "
        "this to work"
    )

pool_of_models = {
      "RandomForestRegressor":RandomForestRegressor,
      "GradientBoostingRegressor":GradientBoostingRegressor,
      "AdaBoostRegressor":AdaBoostRegressor,
      "ExtraTreesRegressor":ExtraTreesRegressor,
      "SVR":SVR,
      "DecisionTreeRegressor":DecisionTreeRegressor,
      "KNeighborsRegressor":KNeighborsRegressor,
      "LinearRegression":LinearRegression,
      "Ridge":Ridge,
      "Lasso":Lasso,
   }

def fix_param_datatypes(param_dict):
    correct_param_dict={}
    for k, v in param_dict.items():
        if v != 'None' and v != 'True' and v != 'False' and v != 'squared_error' and k != 'criterion':
            correct_param_dict[k] = float(v)
        else:
            correct_param_dict[k] = str(v)
    return correct_param_dict

# a function for optimizing the chosen number of models from the model runs
def optimize_model(X_train:pd.DataFrame, y_train:pd.Series, X_val:pd.DataFrame, y_val:pd.Series, params, estimator_name):
   ''' X_train, X_test,y_train, y_test : for training and testing the models with new parameters
    params: chosen paramters from a specific run
   '''
    #estimator_name = tags['estimator_name']
   model_class = pool_of_models[estimator_name]

   param_dict = fix_param_datatypes(params)
   # have a pool of values that will take on the different parameters, a seach space for values for the model
   # assign as value pool space for different parameters where the models will take different parameters from
   
   with mlflow.start_run():

      # start the logging of the experiment
      mlflow.sklearn.autolog()
      # column transformer iss used to transform columns,we are hot encoding 5 columns(the ones that have categorical values) usingg thei indexes
      column_transformer = ColumnTransformer(transformers=[
            ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
      ],remainder='passthrough')
      X_train_tranformed = np.array(column_transformer.fit_transform(X_train))
      X_val_tranformed = np.array(column_transformer.transform(X_val))
      # Use a pipeline to enclose everything to make it easier for deployment
      # use the run paramters given to compoye the best paramters from the space pool
      #params = space_eval(global_pool_space, params)
      # assign those parameters to the model API class
      mlmodel = model_class()

      # fit the model with new parameters
      mlmodel.fit(X_train_tranformed, y_train)
      # do predictions with new parameters
      y_pred = mlmodel.predict(X_val_tranformed)
      # get the rsme and log it
      rmse = mean_squared_error(y_val, y_pred, squared=False)
      r2 = r2_score(y_val,y_pred)
      mlflow.log_metric("r2_score", r2)
      mlflow.log_metric("rmse", rmse)
      print(f"R2 Score : {r2} , validation-rmse : {rmse}")
     
      # mlflow.log_artifact("artifact/preprocessor.b")

# function that will get the previous experiment runs from the other experiment(training) with their parameters,
# from those runs get the top 10 or top 5 runs that have the best rmse and use them for optimizing and getting the best
# paramters for them so they can increate their accuracy
@step(enable_cache = True)
def fine_tune_model(TRAINING_EXPERIMENT_NAME: str, X_train:pd.DataFrame, y_train:pd.DataFrame, X_val:pd.DataFrame, y_val:pd.DataFrame)->Output(
    model_name=str
):
    '''TRAINING_EXPERIMENT_NAME: the previous experiment for training
    log_top_models: number of the top runs to get
     X_train, X_test,y_train, y_test : for training and testing the models with new parameters
    '''

    mlflow.set_experiment(EXPERIMENT_NAME)
    # get the  mlflow client that will enable us to communicate with the backend tracking uri to get the previous runs
    mlflow_client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    # get the experimeent of training  by name
    training_experiment = mlflow_client.get_experiment_by_name(TRAINING_EXPERIMENT_NAME)
    # get the 5 best runs of the previous experiment
    runs = mlflow_client.search_runs(
        experiment_ids=training_experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=5,
        order_by=["metrics.rmse ASC"]
    )
    # from the best top 5 or 10 runs , we get the parameters for that run and pass it to the optimze functon
    for run in runs:
        optimize_model(X_train,y_train,X_val,y_val,run.data.params,run.data.tags['estimator_name'])

    return EXPERIMENT_NAME




