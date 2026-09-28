"""Measures the conventions SF1 needs and writes them to conventions.json:

1. whether MediaPipe's handedness label 'Left' is the signer's own left in the app's landmarks
   (training/cache, made by eval_pretrained.py with the app's extractor);
2. which hand slot of the published KArSL features is the signer's right, by comparing them
   with our own extraction of the same videos;
3. KArSL's effective frame rate, by matching wrist speed with Isharah (25 fps).

    python -m training.sentences.probe_karsl   (needs ~/sli-work/kk/*.npz and ~/sli-work/ish/*.pose)
"""
import glob
import json
import os

import numpy as np

from training import sli_kps
from training.sentences import feats as F

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
CACHE = os.path.join(ROOT, "training", "cache")
KK = os.path.expanduser("~/sli-work/kk")
ISH = os.path.expanduser("~/sli-work/ish")

# (1) app label vs nearest pose wrist
agree = total = 0
for path in sorted(glob.glob(os.path.join(CACHE, "h*_*.json")))[:300]:
    for r in json.load(open(path)):
        if not r["pose"]:
            continue
        for h in r["hands"]:
            w = np.array(h["lms"][0][:2])
            near_left = np.linalg.norm(w - r["pose"][15][:2]) < np.linalg.norm(w - r["pose"][16][:2])
            agree += near_left == (h["label"] == "Left")
            total += 1
left_label_is_left = agree / total > 0.5
print(f"app: label 'Left' is the signer's left in {agree / total:.1%} of {total} hands")

# (2) published npz RH slot vs our RH slot (label 'Right') on the same videos
same = swapped = 0
for z in sorted(glob.glob(os.path.join(KK, "*.npz"))):
    npz = np.load(z)
    for key in npz.files:
        cache = glob.glob(os.path.join(CACHE, f"h*_{key}.json"))
        if not cache:
            continue
        ours = np.stack([sli_kps.features(r) for r in json.load(open(cache[0]))])
        theirs = npz[key]
        n = min(len(ours), len(theirs))
        rh_o, lh_o, rh_t = ours[:n, 142:163, :2], ours[:n, 163:184, :2], theirs[:n, 142:163, :2]
        if not np.abs(rh_t).sum():
            continue
        if np.abs(rh_o - rh_t).mean() < np.abs(lh_o - rh_t).mean():
            same += 1
        else:
            swapped += 1
print(f"npz RH slot = our RH slot in {same} videos, = our LH slot in {swapped}")
# our RH slot holds label 'Right', which is the signer's right iff 'Left' is the signer's left
rh_slot_is_right = (same >= swapped) == left_label_is_left

# (3) frame rate: median wrist speed (shoulder widths per frame), Isharah at 25 fps
from pose_format import Pose


def speed(x):
    w = x[:, 8:12]  # both wrists, relative to the left shoulder
    ok = x[1:, F.O_PRES] * x[:-1, F.O_PRES] > 0
    return np.median(np.abs(np.diff(w, axis=0)).sum(1)[ok])


ish = []
for path in sorted(glob.glob(os.path.join(ISH, "*.pose"))):
    p = Pose.read(open(path, "rb").read())
    d, c = p.body.data[:, 0], p.body.confidence[:, 0]
    w, h = p.header.dimensions.width, p.header.dimensions.height
    ish.append(speed(np.stack([F.from_holistic(d[i], c[i], w, h) for i in range(len(d))])))
ks = []
for z in sorted(glob.glob(os.path.join(KK, "*.npz"))):
    npz = np.load(z)
    for k in npz.files:
        ks.append(speed(np.stack([F.from_karsl184(f) for f in npz[k]])))
ratio = np.median(ish) / max(np.median(ks), 1e-9)
karsl_fps = int(round(25 * ratio / 5) * 5) or 15
print(f"wrist speed per frame: Isharah {np.median(ish):.4f} ({len(ish)} files), KArSL {np.median(ks):.4f} "
      f"({len(ks)} videos) -> KArSL ~ {25 * ratio:.1f} fps, rounded {karsl_fps}")

conv = {"karsl_rh_slot_is_signer_right": bool(rh_slot_is_right), "karsl_fps": karsl_fps,
        "app_left_label_is_signer_left": bool(left_label_is_left), "measured": True}
json.dump(conv, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "conventions.json"), "w"), indent=1)
print(conv)
