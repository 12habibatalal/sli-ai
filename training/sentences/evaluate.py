"""WER of an exported ONNX model on one Isharah split.

    python -m training.sentences.evaluate MODEL.onnx --protocol SI --part dev [--data DIR]
Writes $SLI_OUT/eval-<model>-<protocol>-<part>.json. The SI test split is run once, at the end.
"""
import argparse
import json
import os

import numpy as np
import onnxruntime as ort

from training.sentences import ctc
from training.sentences.train import load_ish
from training.sentences.vocab import encode


def evaluate(onnx_path, feats, index, vocab, protocol, part):
    s = ort.InferenceSession(onnx_path)
    refs, hyps = [], []
    for r in index:
        if r["splits"].get(protocol) != part:
            continue
        lp = s.run(["logprobs"], {"feats": np.asarray(feats[r["id"]], np.float32)[None]})[0][0]
        hyps.append(ctc.greedy(lp)[0])
        refs.append(encode(vocab, r["gloss"]))
    return {"model": os.path.basename(onnx_path), "protocol": protocol, "part": part, "n": len(refs),
            "wer": round(ctc.wer(refs, hyps), 4),
            "sentence_acc": round(float(np.mean([h == r for h, r in zip(hyps, refs)])) if refs else 0.0, 4)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx", nargs="+")
    ap.add_argument("--protocol", default="SI")
    ap.add_argument("--part", default="dev")
    ap.add_argument("--data")
    a = ap.parse_args()
    feats, index, vocab = load_ish(a.data)
    for path in a.onnx:
        if not os.path.exists(path):
            from training.sentences.kaggle_run import find_input
            path = find_input(os.path.basename(path))
        res = evaluate(path, feats, index, vocab, a.protocol, a.part)
        print(res, flush=True)
        out = os.path.join(os.environ.get("SLI_OUT", "."), f"eval-{res['model']}-{a.protocol}-{a.part}.json")
        json.dump(res, open(out, "w"))
