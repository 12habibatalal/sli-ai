"""ArabSign as an external check (different dataset, different people).

Kaggle kernel, two steps:
    python -m training.sentences.arabsign extract      landmarks of the test videos of every
        ArabSign sentence whose words are all in our vocabulary (MediaPipe Holistic, the same
        extractor family as Isharah) -> $SLI_OUT/arabsign_feats.npz + arabsign_index.json
    python -m training.sentences.arabsign eval A.onnx [B.onnx ...]
        WER and sentence accuracy of each model -> $SLI_OUT/eval-<model>-arabsign.json
"""
import glob
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from training.sentences import feats as F
from training.sentences.kaggle_run import find_input

OUT = os.environ.get("SLI_OUT", ".")


def norm(s):
    """Isharah's spelling: bare alef, taa marbuta as haa, alef maqsura as yaa, no diacritics."""
    s = re.sub("[أإآ]", "ا", s).replace("ة", "ه").replace("ى", "ي")
    return re.sub("[ً-ْ]", "", s)


def covered_sentences(vocab):
    import openpyxl

    known = {norm(g): g for g in vocab["glosses"]}
    rows = list(openpyxl.load_workbook(find_input("groundTruth.xlsx")).worksheets[0].iter_rows(values_only=True))[1:]
    out = {}
    for sid, sent in rows:
        toks = [norm(t) for t in str(sent).split()]
        if toks and all(t in known for t in toks):
            out[str(sid).zfill(4)] = {"text": str(sent).strip(), "gloss": " ".join(known[t] for t in toks)}
    return out


def extract_one(path):
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    with mp.solutions.holistic.Holistic(model_complexity=1) as hol:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            r = hol.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            xyz, conf = np.zeros((543, 3)), np.zeros(543)
            for lms, o in ((r.pose_landmarks, 0), (r.face_landmarks, 33), (r.left_hand_landmarks, 501),
                           (r.right_hand_landmarks, 522)):
                if lms:
                    pts = lms.landmark[: 33 if o == 0 else (468 if o == 33 else 21)]
                    xyz[o:o + len(pts)] = [[p.x, p.y, p.z] for p in pts]
                    conf[o:o + len(pts)] = 1
            frames.append(F.from_holistic(xyz, conf, 1.0, 1.0))
    cap.release()
    if not frames:
        return path, None
    t = np.arange(len(frames)) * 1000.0 / fps
    return path, F.resample(np.stack(frames), t).astype(np.float16)


def extract():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "mediapipe==0.10.14", "openpyxl"], check=False)
    vocab = json.load(open(find_input("vocab.json"), encoding="utf-8"))
    sents = covered_sentences(vocab)
    root = os.path.dirname(find_input("groundTruth.xlsx"))  # .../ArabSign/Color
    videos = [p for sid in sents for p in glob.glob(os.path.join(root, "*", "*", "test", sid, "*.mp4"))]
    print(f"{len(sents)} covered sentences, {len(videos)} test videos", flush=True)
    feats, index = {}, []
    with ProcessPoolExecutor() as ex:
        for path, arr in ex.map(extract_one, sorted(videos)):
            if arr is None:
                continue
            key = os.path.basename(path)[:-4]
            sid = os.path.basename(os.path.dirname(path))
            feats[key] = arr
            index.append({"id": key, "sentence": sid, "signer": key[:2], **sents[sid]})
    np.savez(os.path.join(OUT, "arabsign_feats.npz"), **feats)
    json.dump(index, open(os.path.join(OUT, "arabsign_index.json"), "w"), ensure_ascii=False)
    print(f"extracted {len(index)} videos", flush=True)


def evaluate(models):
    import onnxruntime as ort

    from training.sentences import ctc
    from training.sentences.vocab import encode

    vocab = json.load(open(find_input("vocab.json"), encoding="utf-8"))
    z = np.load(find_input("arabsign_feats.npz"))
    index = json.load(open(find_input("arabsign_index.json"), encoding="utf-8"))
    for m in models:
        path = m if os.path.exists(m) else find_input(os.path.basename(m))
        s = ort.InferenceSession(path)
        refs, hyps = [], []
        for r in index:
            lp = s.run(["logprobs"], {"feats": np.asarray(z[r["id"]], np.float32)[None]})[0][0]
            hyps.append(ctc.greedy(lp)[0])
            refs.append(encode(vocab, r["gloss"]))
        res = {"model": os.path.basename(path), "dataset": "ArabSign (test, sentences fully in vocabulary)",
               "sentences": len({r["sentence"] for r in index}), "n": len(refs),
               "signers": len({r["signer"] for r in index}), "wer": round(ctc.wer(refs, hyps), 4),
               "sentence_acc": round(float(np.mean([h == r for h, r in zip(hyps, refs)])), 4)}
        print(res, flush=True)
        json.dump(res, open(os.path.join(OUT, f"eval-{res['model']}-arabsign.json"), "w"), ensure_ascii=False)


if __name__ == "__main__":
    if sys.argv[1] == "extract":
        extract()
    else:
        evaluate(sys.argv[2:])
