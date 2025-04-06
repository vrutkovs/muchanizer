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
    # Try to download the model if it doesn't exist
    if not os.path.exists(args.model_name):
        print(f"Downloading model {args.model_name}...")
        snapshot_download(repo_id=args.model_name)

    model = DiffusersModel(args.model_name)
    # load model from disk
    model.load()
    # start serving loop
    try:
        ModelServer().start([model])
    except Exception:
        print("Quitting...")
