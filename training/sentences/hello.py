"""Kaggle smoke test: GPU, inputs mounted, pip install from the internet works."""
import json
import os
import sys

from training.sentences.kaggle_run import find_input

info = {"python": sys.version}
try:
    import torch

    info.update(torch=torch.__version__, cuda=torch.cuda.is_available(),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
except ImportError:
    info["torch"] = None
info["ann"] = find_input("SI/train.txt")
info["pose"] = find_input("s00_sent0001.pose")
import pose_format  # noqa: E402,F401  (installed by run.py over the internet)

json.dump(info, open(os.path.join(os.environ["SLI_OUT"], "hello.json"), "w"))
print(info)
