from zenml.integrations.deepchecks.steps import (
    DeepchecksDataIntegrityCheckStepParameters,
    deepchecks_data_integrity_check_step
)

Label_column = ""

data_validator = deepchecks_data_integrity_check_step(
    step_name="data_validator",
    params=DeepchecksDataIntegrityCheckStepParameters(
                                                      ),
)

