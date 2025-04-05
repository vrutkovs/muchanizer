#!/usr/bin/env python

try:
    import torch.cuda as tc
    from torch import float16, float32
    from diffusers.schedulers import (DPMSolverMultistepScheduler,
                                      DPMSolverSinglestepScheduler,
                                      EulerDiscreteScheduler,
                                      EulerAncestralDiscreteScheduler,
                                      KDPM2DiscreteScheduler,
                                      HeunDiscreteScheduler,
                                      LMSDiscreteScheduler)
except Exception as e:
    print(f"Caught Exception during library loading: {e}")
    raise e

# size of random operations
RANDOM_BITS_LENGTH = 64

# scheduler types
schedulers = {"DPM++ 2M": DPMSolverMultistepScheduler,
              "DPM++ SDE": DPMSolverSinglestepScheduler,
              "DPM2": KDPM2DiscreteScheduler,
              "Euler a": EulerAncestralDiscreteScheduler,
              "Euler": EulerDiscreteScheduler,
              "Heun": HeunDiscreteScheduler,
              "LMS": LMSDiscreteScheduler}

# check for the presence of a gpu
def get_accelerator_device():
    # assume no gpu is present
    accelerator = "cpu"
    dtype = float32

    # test the presence of a GPU...
    print("Checking for the availability of a GPU...")
    if tc.is_available():
        device_name = tc.get_device_name()
        device_capabilities = tc.get_device_capability()
        device_available_mem, device_total_mem = [x / 1024**3 for x in tc.mem_get_info()]
        print(f"A GPU is available! [{device_name} - {device_capabilities} - {device_available_mem}/{device_total_mem} GB VRAM]")
        accelerator = "cuda"
        dtype = float16

    # return any accelerator found
    return accelerator, dtype
