from steps.feature_selection import seperate_dataset
from steps.feature_selection import label_encode
# from steps.feature_selection import feature_importance
from steps.feature_selection import drop_top_correlated_features
from steps.feature_selection import drop_correlated
from steps.feature_selection import split_for_PCA
from steps.feature_selection import feature_scaling

from pipelines.feature_selection_pipeline import feature_selection_workflow

def main():

    pipeline_instance = feature_selection_workflow(
        seperate_dataset(),
        label_encode(),
        drop_top_correlated_features(),
        drop_correlated(),
        split_for_PCA(),
        feature_scaling()
    )
    pipeline_instance.run(run_name="feature_selection_dagV24", enable_artifact_metadata=True,enable_cache=True)

if __name__=="__main__":
    main()
