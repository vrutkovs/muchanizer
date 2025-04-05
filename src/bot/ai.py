from PIL import ImageFile, Image
from typing import Final
import requests
import os
import json
import base64
import io

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from src import InferRequest
import structlog
log = structlog.get_logger()

VLLM_ENDPOINT: Final = os.getenv("VLLM_ENDPOINT")
if not VLLM_ENDPOINT:
    raise Exception("VLLM_ENDPOINT not set")

VLLM_TOKEN: Final = os.getenv("VLLM_TOKEN")
if not VLLM_TOKEN:
    raise Exception("VLLM_TOKEN not set")

async def img2img_pipeline(image: ImageFile.ImageFile) -> ImageFile.ImageFile:
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')
    image_bytes.seek(0)
    img_b64 = base64.b64encode(image_bytes.read())

    infer_request = InferRequest(
        image_b64=img_b64,
        prompt="cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k",
        negative_prompt="ugly, deformed, disfigured, poor details, bad anatomy",
        num_inference_steps=5,
        width=512,
        height=512,
        guidance_scale=0.7,
    )
    infer_request_json = json.dumps(infer_request)
    response = requests.post(
        VLLM_ENDPOINT,
        data=infer_request_json,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {VLLM_TOKEN}",
        })
    response.raise_for_status()

    response_json = response.json()
    if "predictions" not in response_json:
        raise Exception(f"Unexpected response: {response_json}")
    predictions = response_json["predictions"]

    if len(predictions) == 0 or "image" not in predictions[0]:
        raise Exception(f"Unexpected prediction in response: {predictions}")
    response_image = predictions[0]["image"]

    if "b64" not in response_image:
        raise Exception(f"Unexpected image in response: {response_image}")

    response_imgdata = base64.b64decode(response_image["b64"])
    response_image_obj = Image.open(io.BytesIO(response_imgdata))
    return response_image_obj
