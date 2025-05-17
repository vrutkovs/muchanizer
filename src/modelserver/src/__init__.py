class InferRequest:
    image_b64: str
    prompt: str
    negative_prompt: str
    num_inference_steps: int
    width: int
    height: int
    scheduler: str
    guidance_scale: float
    seed: int

    def __init__(self, image_b64: str, prompt: str, negative_prompt: str,
                 num_inference_steps: int, width: int, height: int, scheduler: str,
                 guidance_scale: float, controlnet_conditioning_scale: float):
        self.image_b64 = image_b64
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.num_inference_steps = num_inference_steps
        self.width = width
        self.height = height
        self.scheduler = scheduler
        self.guidance_scale = guidance_scale
        self.controlnet_conditioning_scale = controlnet_conditioning_scale


class InferResponse:
    image_b64: str

    def __init__(self, image_b64: str):
        self.image_b64 = image_b64
