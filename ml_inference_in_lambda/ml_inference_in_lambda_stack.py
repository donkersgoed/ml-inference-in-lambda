"""Module for the main MlInferenceInLambda Stack."""

# Third party imports
from aws_cdk import (
    core as cdk,
    aws_lambda as lambda_,
    aws_events as eventbridge,
    aws_events_targets as targets,
    aws_iam as iam,
    aws_s3 as s3,
)


class MlInferenceInLambdaStack(cdk.Stack):
    """The MlInferenceInLambda Stack."""

    def __init__(
        self,
        scope: cdk.Construct,
        construct_id: str,
        **kwargs,
    ) -> None:
        """Construct a new MlInferenceInLambdaStack."""
        super().__init__(scope, construct_id, **kwargs)

        # Create an S3 Bucket to store inputs and outputs.
        input_output_bucket = s3.CfnBucket(
            scope=self,
            id="InputOutputBucket",
            notification_configuration=s3.CfnBucket.NotificationConfigurationProperty(
                event_bridge_configuration=s3.CfnBucket.EventBridgeConfigurationProperty(
                    event_bridge_enabled=True
                )
            ),
        )

        # Create the inference function from a Docker image
        inference_function = lambda_.DockerImageFunction(
            scope=self,
            id="InferenceFunction",
            code=lambda_.DockerImageCode.from_image_asset("lambda/functions/inference"),
            environment={
                "MPLCONFIGDIR": "/tmp",
                "GLUON_MODEL": "faster_rcnn_fpn_syncbn_resnest269_coco",
                "DATASET": "b7d778f5",
                "BUCKET": input_output_bucket.ref,
            },
            memory_size=4096,
            timeout=cdk.Duration.minutes(2),
        )
        # Allow the function to read / write to the S3 Bucket
        inference_function.role.add_to_principal_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "s3:*Object*",
                ],
                resources=[cdk.Fn.join("", [input_output_bucket.attr_arn, "/*"])],
            )
        )

        # Set the ephemeral storage size (not supported in L2 constructs yet)
        inference_function_l1: lambda_.CfnFunction = (
            inference_function.node.default_child
        )
        inference_function_l1.add_property_override(
            property_path="EphemeralStorage", value={"Size": 1024}
        )

        # Set a rule to trigger the function for any file uploaded to the inputs/ prefix.
        eventbridge.Rule(
            self,
            "PutObjectRule",
            event_pattern=eventbridge.EventPattern(
                source=["aws.s3"],
                detail_type=["Object Created"],
                detail={"object": {"key": [{"prefix": "inputs/"}]}},
            ),
        ).add_target(targets.LambdaFunction(inference_function))
