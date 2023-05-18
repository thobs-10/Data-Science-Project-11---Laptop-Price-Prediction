from zenml.steps import step, Output, step_output
import numpy as np
import pandas as pd
import mlflow

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from zenml.client import Client

from zenml.integrations.mlflow.experiment_trackers import (
    MLFlowExperimentTracker,
)
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.flavors.mlflow_experiment_tracker_flavor import MLFlowExperimentTrackerSettings

from sklearn.base import RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score,mean_absolute_error

from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor,ExtraTreesRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error, r2_score

TRAINING_EXPERIMENT_NAME = "training-laptop-experiment-1"
mlflow.set_tracking_uri("sqlite:///laptop.db")



experiment_tracker = Client().active_stack.experiment_tracker

if not experiment_tracker or not isinstance(
    experiment_tracker, MLFlowExperimentTracker
):
    raise RuntimeError(
        "Your active stack needs to contain a MLFlow experiment tracker for "
        "this to work"
    )

@step
def train_model(X_train:pd.DataFrame, y_train:pd.DataFrame, X_val:pd.DataFrame, y_val:pd.DataFrame)->Output(
    TRAINING_EXPERIMENT_NAME = str
):
    
    mlflow.set_experiment(TRAINING_EXPERIMENT_NAME)
    mlflow.sklearn.autolog()
    # configured_tracking_uri = get_tracking_uri()
    # print(f"tracking uri from zenml : {configured_tracking_uri}")
    # X_train_transformed = convert_input(X_train)
    # X_val_transformed = convert_input(X_val)
    for model_class in (RandomForestRegressor,GradientBoostingRegressor
                        ,AdaBoostRegressor,
                        ExtraTreesRegressor,SVR,DecisionTreeRegressor,KNeighborsRegressor,
                        LinearRegression,Ridge,Lasso
                        ):
        
        with mlflow.start_run(nested=True):
            #tags of the runs
            mlflow.set_tag("ml-engineer", "thobela")
            # column transformer iss used to transform columns,we are hot encoding 5 columns(the ones that have categorical values) usingg thei indexes
                
            column_transformer = ColumnTransformer(transformers=[
                ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
            ],remainder='passthrough')
            X_train_tranformed = np.array(column_transformer.fit_transform(X_train))
            X_val_tranformed = np.array(column_transformer.transform(X_val))
            # Use a pipeline to enclose everything to make it easier for deployment
            
            mlmodel = model_class()

            
            mlmodel.fit(X_train_tranformed, y_train)
            y_pred = mlmodel.predict(X_val_tranformed)
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            r2 = r2_score(y_val,y_pred)
            # mlflow.log_metric("rmse", rmse)
            # mlflow.log_metric("r2_score", r2)
            print("R2 Score : ",r2)
            print("validation-rmse: ",rmse)
            mlflow.end_run()

    print("\n")
    print('Sucessfully trained...')
    #mlflow_uri = "http://127.0.0.1:4997/"

    return TRAINING_EXPERIMENT_NAME
#C:\Users\Cash Crusaders\Desktop\My Portfolio\Projects\Data Science Projects\Data Science Project 11 - Laptop Price Prediction\dataset\mlruns
#'C:\Users\Cash Crusaders\Desktop\My Portfolio\Projects\Data Science Projects\Data Science Project 11 - Laptop Price Prediction\experimenting\mlruns'
