from steps.feature_engineering import memory_engineering
from steps.feature_engineering import SDD_engineering
from steps.feature_engineering import HDD_engineering
from steps.feature_engineering import drop_columns
from steps.feature_engineering import GPU_engineering
from steps.feature_engineering import apply_cat_os
from steps.feature_engineering import seperate_dataset
from steps.feature_engineering import split_dataset
from steps.feature_engineering import save_dataset


from pipelines.feature_engineering_pipeline import feature_engineering_workflow

import pandas as pd
import numpy as np

def main():

    pipeline_instance = feature_engineering_workflow(
        memory_engineering(),
        SDD_engineering(),
        HDD_engineering(),
        drop_columns(),
        GPU_engineering(),
        apply_cat_os(),
        seperate_dataset(),
        split_dataset(),
        save_dataset()
    )

    pipeline_instance.run(run_name="feature_engineering_dagV5", enable_artifact_metadata=True,enable_cache=True)

if __name__=="__main__":
    main()
