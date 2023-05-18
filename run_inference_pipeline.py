from inferencing.steps.inference_loader import inference_loader
from inferencing.steps.prediction_service_loader import (
    PredictionServiceLoaderStepParameters,
    bentoml_prediction_service_loader,
)
from inferencing.steps.predictor import predictor
from inferencing.steps.prediction_endpoint import preprocess_input_data, make_inference, write_predictions
from constants import MODEL_NAME, PIPELINE_NAME, PIPELINE_STEP_NAME

from inferencing.inference_pipeline.inference_pipeline import inference_laptop_price
def inference_main():
    inference_laptop_price(
            inference_loader(),
            preprocess_input_data(),
            make_inference(),
            write_predictions(),
        ).run(run_name="inferenceV1.3")
    
if __name__ == "__main__":
    inference_main()