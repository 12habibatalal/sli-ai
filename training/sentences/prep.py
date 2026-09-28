"""Kaggle kernel: Isharah-1000 .pose + KArSL keypoints -> SF1 at 15 fps (float16) + vocabulary.

Outputs (in $SLI_OUT): ish_feats.npz, ish_index.json, vocab.json, karsl_feats.npz, karsl_index.json.
    python -m training.sentences.prep [ish] [karsl]      (both when no part is named)
"""
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from training.sentences import feats as F
from training.sentences.kaggle_run import find_input
from training.sentences.vocab import build_vocab

OUT = os.environ.get("SLI_OUT", ".")


def read_split(path):
    rows = {}
    for line in open(path, encoding="utf-8").read().splitlines()[1:]:
        if not line.strip():
            continue
        vid, gloss, text = line.split("|")
        rows[vid.strip()] = (gloss.strip(), text.strip())
    return rows


def ann_id(pose_name):
    """s00_sent0001 (pose file) -> 00_0001 (annotation id)."""
    return pose_name.replace("s", "", 1).replace("_sent", "_")


def ish_one(path):
    from pose_format import Pose

    p = Pose.read(open(path, "rb").read())
    w, h = p.header.dimensions.width, p.header.dimensions.height
    d, c = p.body.data[:, 0], p.body.confidence[:, 0]
    frames = np.stack([F.from_holistic(d[i], c[i], w, h) for i in range(len(d))])
    t = np.arange(len(frames)) * 1000.0 / p.body.fps
    return os.path.basename(path)[:-5], F.resample(frames, t).astype(np.float16)


def karsl_one(path):
    z = np.load(path)
    out = {}
    for k in z.files:
        fr = np.stack([F.from_karsl184(f) for f in z[k]])
        t = np.arange(len(fr)) * 1000.0 / F.CONV["karsl_fps"]
        out[k] = F.resample(fr, t).astype(np.float16)
    return path, out


def isharah():
    ann = os.path.dirname(os.path.dirname(find_input("SI/train.txt")))
    splits = {}
    for proto in ("SI", "US"):
        for part in ("train", "dev", "test"):
            for vid, (g, t) in read_split(os.path.join(ann, proto, f"{part}.txt")).items():
                splits.setdefault(vid, {"gloss": g, "text": t, "splits": {}})["splits"][proto] = part
    pose_dir = os.path.dirname(find_input("s00_sent0001.pose"))
    files = sorted(glob.glob(os.path.join(pose_dir, "*.pose")))
    feats, index = {}, []
    with ProcessPoolExecutor() as ex:
        for vid, arr in ex.map(ish_one, files, chunksize=16):
            key = ann_id(vid)
            if key not in splits:
                continue
            feats[vid] = arr
            index.append({"id": vid, "signer": key[:2], "frames": len(arr), **splits[key]})
    print(f"isharah: {len(files)} pose files, {len(index)} annotated, "
          f"{len(splits) - len(index)} annotations without a pose file", flush=True)
    np.savez(os.path.join(OUT, "ish_feats.npz"), **feats)
    json.dump(index, open(os.path.join(OUT, "ish_index.json"), "w"), ensure_ascii=False)
    vocab = build_vocab(index)
    json.dump(vocab, open(os.path.join(OUT, "vocab.json"), "w"), ensure_ascii=False)
    print(f"vocab: {len(vocab['glosses'])} glosses, {len(vocab['lookup'])} sentences", flush=True)


def karsl():
    kroot = os.path.dirname(os.path.dirname(find_input("karsl-kps/01-test/0001.npz")))
    files = sorted(glob.glob(os.path.join(kroot, "*", "*.npz")))
    kfeats, kindex = {}, []
    with ProcessPoolExecutor() as ex:
        for path, out in ex.map(karsl_one, files, chunksize=8):
            folder = os.path.basename(os.path.dirname(path))  # e.g. 01-train
            sign = int(os.path.basename(path)[:-4])
            for k, arr in out.items():
                key = f"{folder}/{sign:04d}/{k}"
                kfeats[key] = arr
                kindex.append({"key": key, "sign": sign, "signer": folder[:2], "split": folder[3:], "frames": len(arr)})
    print(f"karsl: {len(files)} files, {len(kindex)} videos", flush=True)
    np.savez(os.path.join(OUT, "karsl_feats.npz"), **kfeats)
    json.dump(kindex, open(os.path.join(OUT, "karsl_index.json"), "w"))


if __name__ == "__main__":
    parts = sys.argv[1:] or ["ish", "karsl"]
    if "ish" in parts:
        isharah()
    if "karsl" in parts:
        karsl()
