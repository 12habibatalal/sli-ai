"""Checkpoint -> ONNX (input `feats` [1, T, 356], output `logprobs` [1, ceil(T/2), C]).

    python -m training.sentences.export CKPT OUT.onnx [--quantize] [--latency]
"""
import argparse
import os
import statistics
import time

import numpy as np
import onnxruntime as ort
import torch

from training.sentences import model as M


class _Whole(torch.nn.Module):
    """One unpadded sequence: lengths = T."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, feats):
        lengths = torch.full((1,), feats.shape[1], dtype=torch.long)
        return self.m(feats, lengths)[0]


def export(ckpt, out, quantize):
    c = torch.load(ckpt, map_location="cpu")
    m = M.SignCTC(c["n_classes"], **(c.get("arch") or M.CONFIGS[c["config"]]))
    m.load_state_dict(c["state"])
    m.eval()
    tmp = out + ".fp32.onnx" if quantize else out
    torch.onnx.export(_Whole(m), (torch.zeros(1, 60, 356),), tmp, input_names=["feats"], output_names=["logprobs"],
                      dynamic_axes={"feats": {1: "T"}, "logprobs": {1: "T2"}}, opset_version=17, dynamo=False)
    if quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(tmp, out, weight_type=QuantType.QInt8)
        os.remove(tmp)


def latency(path, frames=300, threads=1):
    """Median seconds for one sequence of `frames` frames on `threads` CPU threads."""
    o = ort.SessionOptions()
    o.intra_op_num_threads = threads
    o.inter_op_num_threads = 1
    s = ort.InferenceSession(path, o)
    x = np.random.default_rng(0).normal(size=(1, frames, 356)).astype(np.float32)
    s.run(None, {"feats": x})
    times = []
    for _ in range(5):
        t = time.perf_counter()
        s.run(None, {"feats": x})
        times.append(time.perf_counter() - t)
    return statistics.median(times)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--quantize", action="store_true")
    ap.add_argument("--latency", action="store_true")
    a = ap.parse_args()
    export(a.ckpt, a.out, a.quantize)
    print(a.out, f"{os.path.getsize(a.out) / 1e6:.1f} MB")
    if a.latency:
        print(f"300 frames, 1 thread: {latency(a.out):.2f} s")
