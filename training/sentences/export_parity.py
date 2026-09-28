"""Fixtures for the browser port: cached KArSL raw frames -> SF1 in Python.

    python -m training.sentences.export_parity          tests/fixtures/sentence-parity.json
    python -m training.sentences.export_parity --model  also tests/fixtures/sentence-model-parity.json
"""
import glob
import json
import os
import sys

import numpy as np

from training.sentences import feats as F

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
CACHE = os.path.join(ROOT, "training", "cache")
files = sorted(glob.glob(os.path.join(CACHE, "h2_*.json")) or glob.glob(os.path.join(CACHE, "h1_*.json")))[:4]
rng = np.random.default_rng(1)
cases = []
for i, path in enumerate(files):
    raws = json.load(open(path))[:40]
    if i == 1:  # hand lost for a stretch and no face (tracking loss)
        for r in raws[10:20]:
            r["hands"] = []
            r["face"] = None
    # cases 0-1 at a steady 30 fps, cases 2-3 with jittery 5-16 fps timestamps
    t = np.cumsum(rng.uniform(60, 200, len(raws))) if i >= 2 else np.arange(len(raws)) * (1000 / 30)
    frames = np.stack([F.from_app_raw(r) for r in raws])
    out = F.resample(frames, t)
    cases.append({"name": os.path.basename(path), "raws": raws, "t": t.tolist(), "frames": out.round(6).tolist()})
json.dump(cases, open(os.path.join(ROOT, "tests", "fixtures", "sentence-parity.json"), "w"))
print(len(cases), "cases")

if "--model" in sys.argv:
    import onnxruntime as ort

    from training.sentences import ctc

    s = ort.InferenceSession(os.path.join(ROOT, "public", "models", "sentences-small.onnx"))
    res = []
    for c in cases:
        lp = s.run(["logprobs"], {"feats": np.array(c["frames"], np.float32)[None]})[0][0]
        res.append({"name": c["name"], "logprobs_head": lp.ravel()[:20].round(5).tolist(), "ids": ctc.greedy(lp)[0]})
    json.dump(res, open(os.path.join(ROOT, "tests", "fixtures", "sentence-model-parity.json"), "w"))
    print(len(res), "model cases")
