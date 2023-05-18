from experimenting.steps.data_loader import ingest_data
from experimenting.steps.data_drift_detector import data_drift_detector
from experimenting.steps.model_training import train_model
from experimenting.steps.tuning import fine_tune_model
from experimenting.steps.model_registering import model_registering

from experimenting.training_pipeline.training_pipeline import training_pipeline

from zenml.integrations.deepchecks.visualizers import DeepchecksVisualizer
from zenml.logger import get_logger

logger = get_logger(__name__)


def main():
    pipeline_instance = training_pipeline(
        ingest_data(),
        train_model(),
        fine_tune_model(),
        model_registering(),
        data_drift_detector,
    )
    pipeline_instance.run(run_name="trainV9.3")

    last_run = pipeline_instance.get_runs()[0]
    data_drift_step = last_run.get_step(step="data_drift_detector")

    DeepchecksVisualizer().visualize(data_drift_step)
    
if __name__ == "__main__":
    main()