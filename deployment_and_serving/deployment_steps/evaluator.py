import numpy as np
import pandas as pd
import pickle
import mlflow
import sklearn
import bentoml
from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from zenml.client import Client
from zenml.integrations.mlflow.experiment_trackers import (
    MLFlowExperimentTracker
)
from zenml.steps import step, Output

from sklearn.metrics import r2_score,mean_squared_error

experiment_tracker = Client().active_stack.experiment_tracker

if not experiment_tracker or not isinstance(
    experiment_tracker, MLFlowExperimentTracker
):
    raise RuntimeError(
        "Your active stack needs to contain an MLFlow experiment tracker for this project to work"
    )


prod_stage="Production"
model_name ="laptop-price-model"

def test_model(model_name, stage, X_val, y_val):
    model = mlflow.sklearn.load_model(f"models:/{model_name}/{stage}")
    y_pred = model.predict(X_val)
    return r2_score(y_val, y_pred), model

@step
def model_evaluator(
    X_train : pd.DataFrame,
    X_val: pd.DataFrame,
    y_val : pd.DataFrame,
)-> Output(
    r2=float,
    model= sklearn.base.RegressorMixin
):
    """Calculate the the R2 of the model """
    mlflow.set_tracking_uri("sqlite:///laptop.db")

    
    model_name = "laptop-price-model"
    # prod_model = mlflow.pyfunc.load_model(f"models:/{model_name}/{prod_stage}")
    # prod_model_r2_score, model = test_model(model_name=model_name, stage=prod_stage, X_val=X_val, y_val=y_val)
    # try -catch exception clause
    column_transformer = ColumnTransformer(transformers=[
            ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
        ],remainder='passthrough')
    X_train_tranformed = column_transformer.fit_transform(X_train)
    X_val_tranformed = column_transformer.transform(X_val)

    logged_model = mlflow.sklearn.load_model(f"models:/{model_name}/{prod_stage}")
    #logged_model = pickle.load(open('./mlruns/1/535f15be557f49bf8de82caede125417/artifacts/model/model.pkl', 'r'))
    prod_model_pred = logged_model.predict(X_val_tranformed)
    logged_model_r2_score = r2_score(y_val,prod_model_pred)
    
    prod_model_r2_score, model = test_model(model_name=model_name, stage="Production", X_val=X_val_tranformed, y_val=y_val)
    if logged_model_r2_score != prod_model_r2_score:
        print("Somethig wrong...check it out.")
    else:
        print("Everything is good, go ahead!")
    
    print(f"Evaluation R2 score : {prod_model_r2_score}")
    # save the model to model to bento for easy deployment process
    saved_model = bentoml.sklearn.save_model("laptop-model", model)
    return prod_model_r2_score, model

    # try:
    #     prod_model = mlflow.pyfunc.load_model(f"models:/{model_name}/{stage}")

    # except mlflow.MlflowException:
    #     print("There is no such model or model version in MLFlow production registry")
    #     # registering the fresh new model from staging to production stage 
    #     client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    #     latest_versions = client.get_latest_versions(name=model_name)

    #     for version in latest_versions:
    #         model_version = version.version
        
    #     new_stage = "Production"
    #     client.transition_model_version_stage(
    #         name=model_name,
    #         version=model_version,
    #         stage=new_stage,
    #         archive_existing_versions=False
    #     )
        
    #     # update the mlflow client
    #     date = datetime.today().date()
    #     client.update_model_version(
    #         name=model_name,
    #         version=model_version,
    #         description=f"The model version {model_version} was transitioned to {new_stage} on {date}"
    #     )
    #     prod_model_r2_score, model = test_model(name=model_name, stage="Production", X_val=X_val_tranformed, y_val=y_val)
    #     return prod_model_r2_score, model
        
    # else:
    #     print("The model is present we can use it")
    #     # production model
    #     prod_r2_score, model = test_model(name=model_name, stage="Production", X_val=X_val_tranformed, y_val=y_val)
    #     # staging Model
    #     staging_r2_score, model = test_model(name=model_name, stage="Staging", X_val=X_val_tranformed, y_val=y_val)
    #     if staging_r2_score < prod_r2_score:
    #         # continue as normal
    #         print("The production model is performing good. Leave it!")
    #         pass
    #     else:
    #         # registering the fresh new model into production
    #         client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    #         latest_versions = client.get_latest_versions(name=model_name)

    #         for version in latest_versions:
    #             model_version = version.version
            
    #         new_stage = "Production"
    #         client.transition_model_version_stage(
    #             name=model_name,
    #             version=model_version,
    #             stage=new_stage,
    #             archive_existing_versions=True
    #         )
            
    #         # update the mlflow client
    #         date = datetime.today().date()
    #         client.update_model_version(
    #             name=model_name,
    #             version=model_version,
    #             description=f"The model version {model_version} was transitioned to {new_stage} on {date}"
    #         )
    #         prod_model_r2_score, model = test_model(name=model_name, stage="Production", X_val=X_val_tranformed, y_val=y_val)
    #     return prod_model_r2_score, model