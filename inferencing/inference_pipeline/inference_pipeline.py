from zenml.config import DockerSettings
from zenml.integrations.constants import BENTOML, PYTORCH
from zenml.pipelines import pipeline

docker_settings = DockerSettings(required_integrations=[PYTORCH, BENTOML])

SERVICE_URL = "http://0.0.0.0:3000/predict"

@pipeline(settings={"docker": docker_settings})
def inference_laptop_price(
    inference_loader,
    preprocess_input_data,
    make_inference,
    write_predictions,
):
    """Link all the steps and artifacts together."""
    input_data, expected_output, X_train = inference_loader()
    preprocessed_data= preprocess_input_data(input_data, X_train)
    predictions = make_inference(preprocessed_data)
    write_predictions(predictions, expected_output)
    