import random

class InferRequest:
    image_b64: str
    prompt: str
    negative_prompt: str
    num_inference_steps: int
    width: int
    height: int
    scheduler: str
    seed: int

    def __init__(self, image_b64: str, prompt: str, negative_prompt: str,
                 num_inference_steps: int, width: int, height: int, scheduler: str):
        self.image_b64 = image_b64
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.num_inference_steps = num_inference_steps
        self.width = width
        self.height = height
        self.scheduler = scheduler

        self.seed = random.getrandbits(64)


class InferResponse:
    image_b64: str

    def __init__(self, image_b64: str):
        self.image_b64 = image_b64
