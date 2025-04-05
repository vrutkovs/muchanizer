#!/usr/bin/env python

# base libs
import os
import base64
import io
from typing import Dict, Union

# import libraries
try:
    import random
    from torch import Generator
    from kserve import Model, InferRequest, InferResponse
    from kserve.errors import InvalidInput
    from diffusers import AutoPipelineForImage2Image
    from .tools import get_accelerator_device, schedulers, RANDOM_BITS_LENGTH
except Exception as e:
    print(f"Caught Exception during library loading: {e}")
    raise e

# stable diffusion class
# instantiate this to perform image generation
class DiffusersModel(Model):
    # initialize class
    def __init__(self, name: str):
        super().__init__(name)
        self.model_id = os.environ.get("MODEL_ID", default="/mnt/models")
        # stable diffusion pipeline
        self.pipeline = None
        # health check
        self.ready = False
        # accelerator device
        self.device = None
        # load model
        self.load()

    # load weights and instantiate pipeline
    def load(self):
        # detect accelerator
        device, dtype = get_accelerator_device()
        try:
            pipeline = AutoPipelineForImage2Image.from_pretrained(self.model_id)
        except Exception:
            # try loading from a single file..
            pipeline = AutoPipelineForImage2Image.from_pretrained(self.model_id, torch_dtype=dtype, use_safetensors=True)

        pipeline.to(device)
        self.pipeline = pipeline
        self.device = device
        # The ready flag is used by model ready endpoint for readiness probes,
        # set to True when model is loaded successfully without exceptions.
        self.ready = True

    # process incoming request payload.
    # An example JSON payload:
    #  {
    #    "instances": [
    #      {
    #        "image_b64": "",
    #        "prompt": "a wizard smoking a pipe",
    #        "negative_prompt": "ugly, deformed, bad anatomy",
    #        "num_inference_steps": 20,
    #        "width": 512,
    #        "height": 512,
    #        "guidance_scale": 7,
    #        "seed": 772847624537827,
    #        "scheduler": "DPM++ 2M",
    #      }
    #    ]
    #  }
    # validate input request: v2 payloads not yet supported
    def preprocess(self, payload: Union[Dict, InferRequest], headers: Dict[str, str] = None) -> Dict:
        if isinstance(payload, Dict) and "instances" in payload:
            headers["request-type"] = "v1"
        # KServe InferRequest not yet supported
        elif isinstance(payload, InferRequest):
            raise InvalidInput("v2 protocol not implemented")
        else:
            # malformed or missing input payload
            raise InvalidInput("invalid payload")

        # return generation data
        return payload["instances"][0]

    # perform a forward pass (inference) and return generated data
    def predict(self, payload: Union[Dict, InferRequest], headers: Dict[str, str] = None) -> Union[Dict, InferResponse]:
        # generate images
        try:
            # set a fixed seed if necessary
            if payload.get("seed") == -1:
                payload["generator"] = Generator(self.device).manual_seed(random.getrandbits(RANDOM_BITS_LENGTH))
            else:
                payload["generator"] = Generator(self.device).manual_seed(payload.get("seed"))

            # Setup Scheduler
            print(f"Generating with Noise Scheduler {payload.get('scheduler')}")
            self.pipeline.scheduler = schedulers.get(payload.get("scheduler")).from_config(self.pipeline.scheduler.config)
            # generate image
            image = self.pipeline(**payload).images[0]
        except Exception:  # error during generation. return random noise
            import numpy as np
            image = np.random.rand(payload.get("width"), payload.get("height"), 3)

        # convert images to PNG and encode in base64
        # for easy sending via response payload
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='PNG')
        image_bytes.seek(0)
        # base64 encoding
        im_b64 = base64.b64encode(image_bytes.read())
        if isinstance(im_b64, bytes):
            im_b64 = im_b64.decode("UTF-8")

        # return payload
        return {
            "predictions": [
                {
                    "model_name": self.model_id,
                    "prompt": payload["prompt"],
                    "negative_prompt": payload.get("negative_prompt", ""),
                    "num_inference_steps": payload.get("num_inference_steps"),
                    "width": payload.get("width", "unspecified"),
                    "height": payload.get("height", "unspecified"),
                    "guidance_scale": payload.get("guidance_scale", "unspecified"),
                    "seed": payload.get("seed", "-1"),
                    "scheduler": payload.get("scheduler", "unspecified"),
                    "image": {
                        "format": "PNG",
                        "b64": im_b64
                    }
                }
            ]}
