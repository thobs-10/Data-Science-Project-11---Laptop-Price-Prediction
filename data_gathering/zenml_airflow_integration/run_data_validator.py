from steps.data_validation import data_validator
from steps.get_data import ingest_data
from steps.feature_engineering import seperate_dataset

from pipelines.data_validator_pipeline import data_validation_pipeline

from zenml.integrations.deepchecks.visualizers import DeepchecksVisualizer
from zenml.logger import get_logger

logger = get_logger(__name__)

def pipeline_run():
    pipeline_instance = data_validation_pipeline(
        ingest_data(),
        seperate_dataset(),
        data_validator = data_validator
    )
    pipeline_instance.run(run_name="data_validator_dagV4",enable_cache=True,enable_artifact_metadata=True)

    last_run = pipeline_instance.get_runs()[0]
    data_val_step = last_run.get_step(step="data_validator")

    DeepchecksVisualizer().visualize(data_val_step)

if __name__=="__main__":
    pipeline_run()