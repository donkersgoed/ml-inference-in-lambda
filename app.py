#!/usr/bin/env python3
"""The main app. Contains all the stacks."""

# Standard library imports
# -

# Third party imports
# -

# Local application/library specific imports
from aws_cdk import core as cdk
from ml_inference_in_lambda.ml_inference_in_lambda_stack import MlInferenceInLambdaStack

app = cdk.App()
MlInferenceInLambdaStack(
    scope=app,
    construct_id="MlInferenceInLambdaStack",
)

app.synth()
