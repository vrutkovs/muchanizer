from PIL import ImageFile, Image
from typing import Final
import requests
import os
import json
import base64
import io

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from request import InferRequest, InferResponse
import structlog
log = structlog.get_logger()

MODEL_ENDPOINT: Final = os.getenv("MODEL_ENDPOINT")
if not MODEL_ENDPOINT:
    raise Exception("MODEL_ENDPOINT not set")

MODEL_TOKEN: Final = os.getenv("MODEL_TOKEN")
HEADERS = {
    "Content-Type": "application/json",
}
if MODEL_TOKEN:
    HEADERS["Authorization"] = f"Bearer {MODEL_TOKEN}"

async def img2img_pipeline(image: ImageFile.ImageFile, prompt: str) -> ImageFile.ImageFile:
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')
    image_bytes.seek(0)
    img_b64 = base64.b64encode(image_bytes.read())

    infer_request = InferRequest(
        image_b64=img_b64.decode("utf-8"),
        prompt=prompt,
        negative_prompt="ugly, deformed, disfigured, poor details, bad anatomy",
        num_inference_steps=20,
        width=1024,
        height=1024,
        guidance_scale=7.5,
        scheduler="DPM++ 2M",
    )
    infer_request_json = json.dumps(
        {"instances": [infer_request]},
        default=lambda o: o.__dict__,
        sort_keys=True,
        indent=4)
    print("---")
    print(f"{infer_request_json}")
    print("---")
    response = requests.post(
        MODEL_ENDPOINT,
        data=infer_request_json,
        headers=HEADERS,
        verify=False)
    response.raise_for_status()
    infer_response = response.json()
    print(infer_response)

    if "predictions" not in infer_response:
        raise Exception("predictions not in response")
    predictions = infer_response["predictions"]
    if len(predictions) == 0:
        raise Exception("no predictions in response")
    first_prediction = predictions[0]
    if "image" not in first_prediction:
        raise Exception("image not in prediction")
    if "b64" not in first_prediction["image"]:
        raise Exception("b64 not in image")

    response_imgdata = base64.b64decode(first_prediction["image"]["b64"])
    return response_imgdata
