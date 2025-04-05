#!/usr/bin/env python

# import base libraries
import argparse

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
    model = DiffusersModel(args.model_name)
    # load model from disk
    model.load()
    # start serving loop
    try:
        ModelServer().start([model])
    except Exception:
        print("Quitting...")
