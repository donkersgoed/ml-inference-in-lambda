"""Inference Lambda Function. Accepts images from S3 and applies object detection on them."""

# Standard library imports
import re
import os
import time

# Third party imports
import boto3
import mxnet as mx
from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt

s3_client = boto3.client("s3")
SUPPORTED_IMAGE_EXTENSIONS_REGEX = r"inputs/.+\.(jpg|jpeg|png)"

GLUON_MODEL = os.environ["GLUON_MODEL"]
BUCKET = os.environ["BUCKET"]
DATASET = os.environ.get("DATASET", True)

LOCAL_INPUT_PATH = "/tmp/input_images"
LOCAL_OUTPUT_PATH = "/tmp/output_images"

os.mkdir(LOCAL_INPUT_PATH)
os.mkdir(LOCAL_OUTPUT_PATH)


def handler(event, _context):  # # pylint: disable=too-many-locals
    """Main Lambda handler."""
    start = time.time()

    print("fetching image from s3")
    object_key: str = event["detail"]["object"]["key"]
    if not re.match(SUPPORTED_IMAGE_EXTENSIONS_REGEX, object_key):
        print(f"unsupported input object: {object_key}")
        return

    _, im_name = os.path.split(object_key)
    local_image_name = f"{LOCAL_INPUT_PATH}/{im_name}"
    s3_client.download_file(BUCKET, object_key, local_image_name)

    time1 = time.time()
    print(f"fetched object in {time1 - start:.2f}s")

    model = model_zoo.get_model(
        GLUON_MODEL,
        pretrained=DATASET,
        root="/tmp",
        ctx=mx.cpu(),
    )

    time2 = time.time()
    print(f"get_model done in {time2 - time1:.2f}s")

    transformed_img, orig_img = data.transforms.presets.rcnn.load_test(local_image_name)

    time3 = time.time()
    print(f"resized image in {time3 - time2:.2f}s")

    # model is asychronous and not actually called here
    box_ids, scores, bboxes = model(transformed_img)

    utils.viz.plot_bbox(  # plot results in inference
        orig_img,
        bboxes[0],
        scores[0],
        box_ids[0],
        thresh=0.7,
        class_names=model.classes,
        linewidth=1,
    )

    time4 = time.time()
    print(f"inference and plot completed in {time4 - time3:.2f}s")

    end = time.time()
    print(f"finished in {end - start:.2f}s")

    local_output_name = f"/{LOCAL_OUTPUT_PATH}/{im_name}"
    plt.savefig(local_output_name)
    s3_client.upload_file(local_output_name, BUCKET, f"outputs/{im_name}")


if __name__ == "__main__":
    handler({}, None)
