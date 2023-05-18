import numpy as np
import pandas as pd
from zenml.steps import step, Output, step_output
@step
def deploy_demo()->None:
    print("hello this is deployment demo")