from zenml.config import DockerSettings
from zenml.integrations.constants import DEEPCHECKS, SKLEARN
from zenml.pipelines import pipeline

docker_settings = DockerSettings(required_integrations=[DEEPCHECKS, SKLEARN])

@pipeline(enable_cache=True, settings={"docker":docker_settings})
def data_validation_pipeline(get_data, seperate_dataset, data_validator):
    df = get_data()
    X , y = seperate_dataset(df)
    data_validator(dataset = X)
