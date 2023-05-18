
from steps.get_data import ingest_data 
from steps.preprocessing import fix_column_datatypes
from steps.preprocessing import create_column
from steps.preprocessing import preprocess_features
from steps.preprocessing import drop_unwanted_columns
from steps.preprocessing import apply_fetch_processors

#from steps.feature_engineering import memory_engineering
from zenml.integrations.airflow.orchestrators.airflow_orchestrator import AirflowOrchestrator

from pipelines.preprocessing_pipeline import preprocessing_workflow
#from pipelines.feature_engineering import feature_engineering_workflow

from zenml.post_execution import get_run

import pandas as pd

# read the data from the dataset folder
df1 = pd.DataFrame()
def main():
    pipeline_instance = preprocessing_workflow(
        ingest_data(),
        fix_column_datatypes(),
        create_column(),
        preprocess_features(),
        drop_unwanted_columns(),
        apply_fetch_processors()
    )

    pipeline_instance.run(run_name='preprocessing_pipeline_dagV8',enable_cache=True,enable_artifact_metadata=True)
  
if __name__ == "__main__":
    main()
    
