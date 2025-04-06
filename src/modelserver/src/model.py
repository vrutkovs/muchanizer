#!/usr/bin/env python

# import base libraries
import argparse
import os
from huggingface_hub import snapshot_download

# import custom diffuser class
# also import kserve libraries
try:
    from libs.diffuser_class import DiffusersModel
    from kserve import ModelServer, model_server
except Exception as e:
    print(f"Caught Exception during library loading: {e}")
    exit(-1)

# generate automatic argument parser
parser = argparse.ArgumentParser(parents=[model_server.parser])
args, _ = parser.parse_known_args()

# start serving
if __name__ == "__main__":
    refiner_model = os.environ.get("REFINER_MODEL", None)
    vae_model = os.environ.get("VAE_MODEL", None)
    lora_model = os.environ.get("LORA_MODEL", None)
    lora_weight_name = os.environ.get("LORA_WEIGHT_NAME", None)

    # Try to download the model if it doesn't exist
    if not os.path.exists(args.model_name):
        model = os.environ.get("MODEL_ID", None)
        if model:
            print(f"Downloading model {model}...")
            snapshot_download(repo_id=model)

        if vae_model:
            print(f"Downloading VAE model {vae_model}...")
            snapshot_download(repo_id=vae_model)

        if refiner_model:
            print(f"Downloading refiner model {refiner_model}...")
            snapshot_download(repo_id=refiner_model)

        if lora_model:
            print(f"Downloading lora model {lora_model}...")
            snapshot_download(repo_id=lora_model)

        print("All models downloaded.")

    model = DiffusersModel(args.model_name)
    # start serving loop
    try:
        ModelServer().start([model])
    except Exception:
        print("Quitting...")
