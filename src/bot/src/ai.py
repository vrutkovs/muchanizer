from PIL import ImageFile, Image
from typing import Final
import requests
import os
import json
import base64
import io

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from request import InferRequest,InferResponse
import structlog
log = structlog.get_logger()

MODEL_ENDPOINT: Final = os.getenv("MODEL_ENDPOINT")
if not MODEL_ENDPOINT:
    raise Exception("MODEL_ENDPOINT not set")

MODEL_TOKEN: Final = os.getenv("MODEL_TOKEN")
if not MODEL_TOKEN:
    raise Exception("MODEL_TOKEN not set")

async def img2img_pipeline(image: ImageFile.ImageFile) -> ImageFile.ImageFile:
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')
    image_bytes.seek(0)
    img_b64 = base64.b64encode(image_bytes.read())

    infer_request = InferRequest(
        image_b64=str(img_b64),
        prompt="cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k",
        negative_prompt="ugly, deformed, disfigured, poor details, bad anatomy",
        num_inference_steps=5,
        width=512,
        height=512,
    )
    infer_request_json = json.dumps(infer_request, default=lambda o: o.__dict__, sort_keys=True, indent=4)
    response = requests.post(
        MODEL_ENDPOINT,
        data=infer_request_json,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {MODEL_TOKEN}",
        })
    response.raise_for_status()

    response_json = response.json()
    infer_response = response_json.loads(response_json, object_hook=lambda d: InferResponse(**d))

    response_imgdata = base64.b64decode(infer_response.image_b64["b64"])
    response_image_obj = Image.open(io.BytesIO(response_imgdata))
    return response_image_obj
