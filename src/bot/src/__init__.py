class InferRequest:
    image_b64: bytes
    prompt: str
    negative_prompt: str
    num_inference_steps: int
    width: int
    height: int
    guidance_scale: float

    def __init__(self, image_b64: bytes, prompt: str, negative_prompt: str,
                 num_inference_steps: int, width: int, height: int,
                 guidance_scale: float, strength: float):
        self.image_b64 = image_b64
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.num_inference_steps = num_inference_steps
        self.width = width
        self.height = height
        self.guidance_scale = guidance_scale
        self.strength = strength


class InferResponse:
    image_b64: str

    def __init__(self, image_b64: str):
        self.image_b64 = image_b64
