#!/usr/bin/env python

# base libs
import os
import base64
import io
from typing import Dict, Union

# import libraries
import random
from torch import Generator
from kserve import Model, InferRequest, InferResponse
from kserve.errors import InvalidInput
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.pipelines.auto_pipeline import AutoPipelineForImage2Image
from .tools import get_accelerator_device, schedulers, RANDOM_BITS_LENGTH
from PIL import Image
# from diffusers.utils.pil_utils import pt_to_pil

# Torch tweaks
import torch
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True

# stable diffusion class
# instantiate this to perform image generation
class DiffusersModel(Model):
    # initialize class
    def __init__(self, name: str, refiner_model: str | None = None, vae_model: str | None = None, lora_model: str | None = None):
        super().__init__(name)
        self.model_id = os.environ.get("MODEL_ID", default="/mnt/models")
        # stable diffusion pipeline
        self.pipeline = None
        # health check
        self.ready = False
        # accelerator device
        self.device = None
        # refiner
        self.refiner_model = refiner_model
        self.refiner = None
        # vae
        self.vae_model = vae_model
        # lora
        self.lora_model = lora_model
        # load model
        self.load()

    # load weights and instantiate pipeline
    def load(self):
        print(f"Loading model {self.model_id}")
        # detect accelerator
        device, dtype = get_accelerator_device()
        try:
            pipeline = AutoPipelineForImage2Image.from_pretrained(self.model_id)
        except Exception:
            # try loading from a single file..
            vae = None
            if self.vae_model:
                print(f"Loading VAE {self.vae_model}")
                vae = AutoencoderKL.from_pretrained(self.vae_model, torch_dtype=dtype)

            pipeline = AutoPipelineForImage2Image.from_pretrained(self.model_id, vae=vae, torch_dtype=dtype, variant="fp16", use_safetensors=True)

            if self.lora_model:
                print(f"Loading LoRA {self.lora_model}")
                pipeline.load_lora_weights(self.lora_model)

            if self.refiner_model:
                print(f"Loading refiner {self.refiner_model}")
                self.refiner = AutoPipelineForImage2Image.from_pretrained(self.refiner_model, vae=vae, torch_dtype=dtype, variant="fp16", use_safetensors=True, text_encoder_2=pipeline.text_encoder_2)
                self.refiner.to(device)

        pipeline.enable_attention_slicing()
        pipeline.unet.to(memory_format=torch.channels_last)
        pipeline.vae.to(memory_format=torch.channels_last)
        pipeline.fuse_qkv_projections()

        # Max autotune
        # pipeline.unet = torch.compile(pipeline.unet, mode="max-autotune", fullgraph=True)
        # pipeline.vae.decode = torch.compile(pipeline.vae.decode, mode="max-autotune", fullgraph=True)

        # Reduce overhead
        # pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=True)
        # pipeline.vae.decode = torch.compile(pipeline.vae.decode, mode="reduce-overhead", fullgraph=True)

        pipeline.to(device)

        self.pipeline = pipeline
        self.device = device
        # The ready flag is used by model ready endpoint for readiness probes,
        # set to True when model is loaded successfully without exceptions.
        print("Loading complete")
        self.ready = True
        return True

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
        print("payload:")
        print(payload)
        print("---")

        if isinstance(payload, Dict) and "instances" in payload:
            headers["request-type"] = "v1"
        # KServe InferRequest not yet supported
        elif isinstance(payload, InferRequest):
            raise InvalidInput("v2 protocol not implemented")
        else:
            # malformed or missing input payload
            raise InvalidInput("invalid payload")

        if len(payload["instances"]) == 0:
            raise InvalidInput("invalid payload - empty instances")

        if "image_b64" not in payload["instances"][0]:
            raise InvalidInput("invalid payload - image_b64 not set")

        # return generation data
        return payload["instances"][0]

    # perform a forward pass (inference) and return generated data
    def predict(self, payload: Union[Dict, InferRequest], headers: Dict[str, str] = None, response_headers: Dict[str, str] = None,) -> Union[Dict, InferResponse]:
        # generate images
        # set a fixed seed if necessary
        if payload.get("seed", -1) == -1:
            payload["generator"] = Generator(self.device).manual_seed(random.getrandbits(RANDOM_BITS_LENGTH))
        else:
            payload["generator"] = Generator(self.device).manual_seed(payload.get("seed"))

        # Setup Scheduler
        print(f"Generating with Noise Scheduler {payload.get('scheduler')}")
        self.pipeline.scheduler = schedulers.get(payload.get("scheduler")).from_config(self.pipeline.scheduler.config)

        # Convert base64 encoded image to PIL Image
        image_b64 = payload.get("image_b64")
        image = Image.open(io.BytesIO(base64.b64decode(image_b64)))
        payload["image"] = image

        # generate image
        print(f"Params: {payload}")
        tensor = self.pipeline(**payload, denoising_end=0.8, output_type="latent").images
        if self.refiner:
            tensor = self.refiner(**payload, denoising_start=0.8, image=tensor).images[0]
        # Convert tensor to PIL Image
        # image = pt_to_pil(tensor[0])
        image = tensor[0]

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
