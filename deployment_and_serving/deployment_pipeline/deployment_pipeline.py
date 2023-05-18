from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW, SKLEARN, BENTOML
from zenml.pipelines import pipeline

docker_settings = DockerSettings(required_integrations=[MLFLOW, SKLEARN, BENTOML])
pipeline_name="continous_deployment"
step_name = "model_deployer"
model_name = "laptop-price-model"
running = True
run_name = "wistful-croc-520"
run_id = "535f15be557f49bf8de82caede125417"
experiment_name = "hyper-optimization-experiment-1"
model_name = "laptop-price-model"

@pipeline(enable_cache=False, enable_artifact_metadata=False, settings={"docker":docker_settings} ,name="deployment_pipeline")
def continous_deployment(
    ingest_data,
    model_evaluator,
    deployment_trigger,
    bento_builder,
    deployer,
    # deploy_demo,
    # model_deployer,
    # deploy_model_step
):
    X_train,X_test,y_test,y_train = ingest_data()
    evaluated_r2, model = model_evaluator(X_train,X_test,y_test)
    deployment_decision = deployment_trigger(evaluated_r2)
    bento = bento_builder(model=model)
    deployer(deploy_decision=deployment_decision, bento=bento)
    # deploy_demo()
    # service = deploy_model(deployment_decision, model)
    # model_deployer(pipeline_name,step_name,model_name, running, deployment_decision)
    # model_deployer(deployment_decision, model)
    #predictor(existing_services,)

