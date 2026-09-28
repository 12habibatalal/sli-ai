# Sentence Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recognise continuous Saudi Sign Language sentences from unseen signers with a large
server model and a distilled ≤ 20 MB phone model, ship both in the SLI app and on Hugging Face.

**Architecture:** Landmarks (MediaPipe) → "sentence features v1" (SF1, 356 values/frame at
15 fps) → Conformer-lite encoder + CTC over Isharah glosses. Training runs on Kaggle GPUs,
driven from the pricelens server with the Kaggle CLI. The large model is served by a small
ONNX Runtime HTTP service behind `pricelens-proxy`; the phone model runs in a web worker and
is the fallback.

**Tech Stack:** Python 3.12 (numpy, PyTorch, onnx, onnxruntime, pose-format, pytest, kaggle
CLI 2.x), TypeScript/Vite, onnxruntime-web 1.30, vitest, Playwright, Docker/Podman, nginx.

**Spec:** `docs/superpowers/specs/2026-09-28-sentence-model-design.md`

## Global Constraints

- Everything happens in `~/sli-ai` on the pricelens server (`ssh pricelens`). The phone copy is only an editing mirror.
- Python tooling: `~/sli-work/kvenv` (Python 3.12; `uv pip install -p ~/sli-work/kvenv ...`). The old `~/sli-work/venv` (3.9, mediapipe 0.10.14) stays for the existing scripts.
- Long ssh commands drop: run them with `setsid nohup ... > log 2>&1 &` and poll the log. Never `pkill -f` a pattern that matches your own ssh command.
- Nothing heavy runs on pricelens. Anything taking more than about 10 CPU-minutes runs on Kaggle under account `baraasaad` (token in `~/.kaggle/access_token`).
- Licence of the models and derived artefacts: **CC-BY-NC-SA-4.0**. Publish weights and metadata only, never videos or landmarks.
- SF1: 356 float32 values per frame (176 points × (x, y) + 4 presence flags), 15 fps, blank = class 0.
- Phone model ONNX ≤ 20 MB. Large model: one 20 s sentence (300 frames) must decode in ≤ 2 s on one server core.
- API: `POST /api/sentence`, body float32 little-endian, ≤ 450 frames (641 KB), container limited to 1 CPU, published on `127.0.0.1:3021`. The proxy vhost points at `127.0.0.1:<port>`, never at a container name.
- The Isharah SI test split is evaluated **once**, in Task 12. Every selection decision uses SI dev.
- Commit messages end with the two attribution lines:
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01BHnuerFvc7H93UCpCYsVDN`.
  Do not push to GitHub without asking the user.

## Review Focus

1. **A hand leaves the picture mid-sentence, or the face is not found.** The recognizer keeps decoding with the presence flag at 0, with no NaN and no crash. Tests go in Task 2 (TS features) and Task 13 (server all-zero frames).
2. **Very short movement** (under 600 ms, e.g. scratching the nose). Nothing is sent to the server and no sentence is produced. Test goes in Task 15.
3. **Signing longer than 30 s.** The recognizer finalises at 30 s, and the server answers 413 above 450 frames. Tests go in Task 13 and Task 15.
4. **Server slow, down or offline.** The phone model answers within 4 s plus its own inference time, and the result is labelled as coming from the phone. Test goes in Task 15.
5. **Slow or irregular frame rate** (5–10 fps, jittery timestamps). Resampling to 15 fps gives the same result in Python and TS. Tests go in Task 1 and Task 2.

---

## File map

```
training/sentences/
  __init__.py
  feats.py            SF1 definition: from_holistic / from_app_raw / from_karsl184 / resample / mirror
  conventions.json    measured constants (Task 3)
  probe_karsl.py      measures KArSL hand-slot convention and frame rate (Task 3)
  export_parity.py    Python→TS fixtures (Task 2, Task 11)
  kaggle_run.py       bundle repo → Kaggle kernel, push, poll, pull (Task 4)
  prep.py             Kaggle kernel: Isharah + KArSL → 15 fps SF1 shards + vocab (Task 5)
  vocab.py            gloss vocabulary + gloss→text lookup (Task 5)
  ctc.py              greedy decode, WER (Task 6)
  model.py            encoder + CTC head, configs "large"/"small" (Task 6)
  data.py             dataset, augmentation, batching (Task 7)
  train.py            stage1 / stage2 / distill entry point (Task 8, 10)
  export.py           ONNX export, int8 quantisation, parity check (Task 11)
  evaluate.py         WER on a split with an ONNX model (Task 12)
  arabsign.py         Kaggle kernel: ArabSign landmarks + overlap eval (Task 12)
  rounds.md           training log: one row per round (Task 9)
  tests/              pytest
server/sentence-api/
  app.py  requirements.txt  Dockerfile  run.sh  test_app.py   (Task 13)
src/recognition/
  sentenceFeatures.ts  ctc.ts  sentenceModel.ts  sentence.worker.ts  sentenceClient.ts  sentences.ts
  engine.ts, interpreter.ts (modified)
src/ui/capture.ts, src/pages/sign.ts, src/i18n.ts (modified)
public/models/sentences-small.onnx, public/models/sentences-vocab.json
tests/unit/sentenceFeatures.test.ts, ctc.test.ts, sentences.test.ts
tests/replay/sentences.test.ts, tests/e2e/sentences.spec.ts
hf/README.md, hf/upload.py
~/pricelens/docker/nginx-upstreams/sli.conf (server file, not in this repo)
```

---

### Task 1: SF1 features in Python

**Files:**
- Create: `training/sentences/__init__.py` (empty), `training/sentences/feats.py`, `training/sentences/tests/test_feats.py`, `training/sentences/conventions.json`
- Test: `training/sentences/tests/test_feats.py`

**Interfaces:**
- Produces:
  - `feats.FEAT_DIM = 356`, `feats.FPS = 15`, `feats.N_POINTS = 176`
  - `from_holistic(xyz: np.ndarray[543,3], conf: np.ndarray[543], w: float, h: float) -> np.ndarray[356]`
  - `from_app_raw(raw: dict) -> np.ndarray[356]`
  - `from_karsl184(f: np.ndarray[184,4]) -> np.ndarray[356]`
  - `resample(frames: np.ndarray[T,356], t_ms: np.ndarray[T], fps=FPS) -> np.ndarray[T',356]`
  - `mirror(frames: np.ndarray[T,356]) -> np.ndarray[T,356]`

Layout of one frame: `[pose 6×2 | face 128×2 | left hand 21×2 | right hand 21×2 | presence pose, face, lh, rh]`.
Offsets: pose 0, face 12, left hand 268, right hand 310, presence 352.

- [ ] **Step 1: Write the failing tests**

```python
# training/sentences/tests/test_feats.py
import numpy as np
import pytest
from training.sentences import feats as F

rng = np.random.default_rng(0)


def fake_raw(with_pose=True, with_face=True, hands=("Left", "Right")):
    pose = rng.random((33, 4)).tolist() if with_pose else None
    face = rng.random((478, 3)).tolist() if with_face else None
    hs = []
    for lab in hands:
        lms = rng.random((21, 3))
        hs.append({"label": lab, "lms": lms.tolist()})
    return {"pose": pose, "face": face, "hands": hs}


def test_layout_and_presence():
    f = F.from_app_raw(fake_raw())
    assert f.shape == (F.FEAT_DIM,) and F.FEAT_DIM == 356
    assert list(f[352:]) == [1, 1, 1, 1]
    empty = F.from_app_raw({"pose": None, "face": None, "hands": []})
    assert not empty.any()


def test_pose_anchor_and_scale():
    raw = fake_raw()
    f = F.from_app_raw(raw)
    p = np.array(raw["pose"])
    w = np.hypot(*(p[11, :2] - p[12, :2]))
    assert np.allclose(f[0:2], 0)  # left shoulder is the origin
    assert np.allclose(f[2:4], (p[12, :2] - p[11, :2]) / w, atol=1e-6)


def test_hands_go_to_nearest_wrist_not_label():
    raw = fake_raw()
    pose = np.array(raw["pose"])
    # put the hand labelled "Left" at the signer's RIGHT wrist (pose 16)
    lh = np.array(raw["hands"][0]["lms"]); lh[:, :2] += pose[16, :2] - lh[0, :2]
    rh = np.array(raw["hands"][1]["lms"]); rh[:, :2] += pose[15, :2] - rh[0, :2]
    raw["hands"][0]["lms"], raw["hands"][1]["lms"] = lh.tolist(), rh.tolist()
    f = F.from_app_raw(raw)
    scale = np.hypot(*(lh[2, :2] - lh[17, :2]))
    assert np.allclose(f[310:352].reshape(21, 2), (lh[:, :2] - lh[0, :2]) / scale, atol=1e-6)


def test_holistic_matches_app_raw():
    raw = fake_raw()
    pose = np.array(raw["pose"])
    for h, wrist in zip(raw["hands"], (15, 16)):  # make nearest-wrist agree with holistic slots
        lms = np.array(h["lms"]); lms[:, :2] += pose[wrist, :2] - lms[0, :2]; h["lms"] = lms.tolist()
    xyz = np.zeros((543, 3)); conf = np.zeros(543)
    xyz[:33] = pose[:, :3] * [1000, 1000, 1]; conf[:33] = 1
    xyz[33:501] = np.array(raw["face"])[:468] * [1000, 1000, 1]; conf[33:501] = 1
    xyz[501:522] = np.array(raw["hands"][0]["lms"]) * [1000, 1000, 1]; conf[501:522] = 1
    xyz[522:543] = np.array(raw["hands"][1]["lms"]) * [1000, 1000, 1]; conf[522:543] = 1
    assert np.allclose(F.from_holistic(xyz, conf, 1000, 1000), F.from_app_raw(raw), atol=1e-5)


def test_karsl184_face_rescale_is_exact():
    from training import sli_kps
    raw = fake_raw(hands=())
    raw184 = {"pose": raw["pose"], "face": raw["face"], "hands": []}
    f184 = sli_kps.features(raw184)
    a = F.from_karsl184(f184)
    b = F.from_app_raw(raw)
    assert np.allclose(a[:268], b[:268], atol=1e-5)


def test_resample_irregular_timestamps():
    t = np.array([0, 90, 210, 260, 400, 530], dtype=float)  # ~7-11 fps with jitter
    x = np.stack([np.full(F.FEAT_DIM, v) for v in t])  # value == time, so linear interp is exact
    out = F.resample(x, t)
    assert out.shape[0] == int(530 / (1000 / 15)) + 1
    assert np.allclose(out[:, 0], np.arange(out.shape[0]) * 1000 / 15)


def test_mirror_twice_is_identity_and_swaps_hands():
    f = np.stack([F.from_app_raw(fake_raw()) for _ in range(3)])
    m = F.mirror(f)
    assert np.allclose(F.mirror(m), f, atol=1e-6)
    assert np.allclose(m[:, 268:310:2], -f[:, 310:352:2])  # left x = -(right x)
    assert np.allclose(m[:, 352:], f[:, [352, 353, 355, 354]])
```

- [ ] **Step 2: Run the tests to check they fail**

Run: `cd ~/sli-ai && ~/sli-work/kvenv/bin/python -m pytest training/sentences/tests/test_feats.py -q`
(first: `~/.local/bin/uv pip install -q -p ~/sli-work/kvenv pytest numpy mediapipe==0.10.14 || ~/.local/bin/uv pip install -q -p ~/sli-work/kvenv pytest numpy`. If mediapipe has no 3.12 wheel, `test_karsl184_face_rescale_is_exact` imports only `sli_kps.features`, so make that import lazy as described in Step 3.)
Expected: FAIL with `ModuleNotFoundError: training.sentences`.

- [ ] **Step 3: Implement**

`training/sli_kps.py` imports mediapipe at module top. To use `features()` without mediapipe, move `import mediapipe as mp` and `from mediapipe.tasks.python import BaseOptions, vision` inside `Extractor.__init__` and `Extractor.raw`. Behaviour is unchanged. Also create `training/__init__.py` (empty) if it is missing.

```python
# training/sentences/conventions.json   (placeholder values are overwritten by Task 3)
{"karsl_rh_slot_is_signer_right": true, "karsl_fps": 15, "app_left_label_is_signer_left": false, "measured": false}
```

```python
# training/sentences/feats.py
"""Sentence features v1 (SF1): 176 points x (x, y) + 4 presence flags per frame, 15 fps.

Layout: pose 6 | face 128 | left hand 21 | right hand 21 | presence (pose, face, lh, rh).
Hands are stored by the signer's own side. src/recognition/sentenceFeatures.ts is the browser
port; tests/unit/sentenceFeatures.test.ts checks parity against export_parity.py.
"""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FACE_IDX = json.load(open(os.path.join(HERE, "..", "face_idx.json")))["face"][:128]  # no iris
CONV = json.load(open(os.path.join(HERE, "conventions.json")))

POSE_IDX = (11, 12, 13, 14, 15, 16)
N_POSE, N_FACE, N_HAND = 6, 128, 21
N_POINTS = N_POSE + N_FACE + 2 * N_HAND  # 176
FEAT_DIM = N_POINTS * 2 + 4  # 356
FPS = 15
O_POSE, O_FACE, O_LH, O_RH, O_PRES = 0, 12, 268, 310, 352
FACE_ANCHOR = 1  # subset position 1 = mesh point 7, as the word model
FACE_SCALE = (33, 263)  # outer eye corners (mesh indices)
HAND_SCALE = (2, 17)
SWAP = [1, 0, 3, 2, 5, 4]  # pose subset left<->right


def _d(a, b):
    return max(float(np.hypot(a[0] - b[0], a[1] - b[1])), 1e-6)


def _pose(out, p):  # p: (33, >=2) in image-normalised coords
    sub = p[list(POSE_IDX), :2]
    out[O_POSE:O_POSE + 12] = ((sub - sub[0]) / _d(p[11], p[12])).ravel()
    out[O_PRES] = 1


def _face(out, f):  # f: (>=468, >=2)
    sub = f[FACE_IDX, :2]
    out[O_FACE:O_FACE + 256] = ((sub - sub[FACE_ANCHOR]) / _d(f[FACE_SCALE[0]], f[FACE_SCALE[1]])).ravel()
    out[O_PRES + 1] = 1


def _hand(out, h, left):  # h: (21, >=2)
    o = O_LH if left else O_RH
    out[o:o + 42] = ((h[:, :2] - h[0, :2]) / _d(h[HAND_SCALE[0]], h[HAND_SCALE[1]])).ravel()
    out[O_PRES + (2 if left else 3)] = 1


def from_holistic(xyz, conf, w, h):
    """MediaPipe Holistic frame as stored by pose-format: 33 pose, 468 face, 21 left, 21 right."""
    out = np.zeros(FEAT_DIM, np.float32)
    p = np.asarray(xyz, np.float64)[:, :2] / [w, h]
    c = np.asarray(conf)
    if c[11] > 0 and c[12] > 0:
        _pose(out, p[:33])
    if c[33:501].any():
        _face(out, p[33:501])
    if c[501:522].any():
        _hand(out, p[501:522], True)
    if c[522:543].any():
        _hand(out, p[522:543], False)
    return out


def _assign(pose, hands):
    """Signer side for each hand: nearest pose wrist (15 = left, 16 = right)."""
    if pose is None:
        flip = CONV["app_left_label_is_signer_left"]
        return [(h["label"] == "Left") == flip for h in hands]
    lw, rw = np.asarray(pose[15][:2]), np.asarray(pose[16][:2])
    wr = [np.asarray(h["lms"][0][:2]) for h in hands]
    if len(hands) == 2:
        straight = np.linalg.norm(wr[0] - lw) + np.linalg.norm(wr[1] - rw)
        crossed = np.linalg.norm(wr[0] - rw) + np.linalg.norm(wr[1] - lw)
        return [True, False] if straight <= crossed else [False, True]
    return [np.linalg.norm(w - lw) <= np.linalg.norm(w - rw) for w in wr]


def from_app_raw(raw):
    """RawFrame as produced by the app / sli_kps.Extractor.raw."""
    out = np.zeros(FEAT_DIM, np.float32)
    if raw["pose"]:
        _pose(out, np.asarray(raw["pose"], np.float64))
    if raw["face"]:
        _face(out, np.asarray(raw["face"], np.float64))
    hands = raw["hands"][:2]
    for hand, left in zip(hands, _assign(raw["pose"], hands)):
        _hand(out, np.asarray(hand["lms"], np.float64), left)
    return out


def from_karsl184(f):
    """The word model's (184, 4) features (as published for KArSL) -> SF1."""
    f = np.asarray(f, np.float64)
    out = np.zeros(FEAT_DIM, np.float32)
    pose, face, rh, lh = f[0:6, :2], f[6:134, :2], f[142:163, :2], f[163:184, :2]
    if np.abs(pose).sum() > 0:
        out[O_POSE:O_POSE + 12] = pose.ravel()  # same anchor and scale as SF1
        out[O_PRES] = 1
    if np.abs(face).sum() > 0:
        i33, i263 = FACE_IDX.index(33), FACE_IDX.index(263)
        out[O_FACE:O_FACE + 256] = (face / _d(face[i33], face[i263])).ravel()
        out[O_PRES + 1] = 1
    right_is_right = CONV["karsl_rh_slot_is_signer_right"]
    for slot, is_rh_slot in ((rh, True), (lh, False)):
        if np.abs(slot).sum() == 0:
            continue
        left = (not is_rh_slot) if right_is_right else is_rh_slot
        o = O_LH if left else O_RH
        out[o:o + 42] = slot.ravel()  # already wrist-relative, hand-scaled
        out[O_PRES + (2 if left else 3)] = 1
    return out


def resample(frames, t_ms, fps=FPS):
    """Linear interpolation of frames taken at t_ms onto a uniform fps grid starting at t_ms[0]."""
    frames = np.asarray(frames, np.float32)
    t = np.asarray(t_ms, np.float64) - t_ms[0]
    step = 1000.0 / fps
    grid = np.arange(int(t[-1] / step) + 1) * step
    idx = np.clip(np.searchsorted(t, grid, side="right") - 1, 0, len(t) - 2) if len(t) > 1 else np.zeros(len(grid), int)
    if len(t) == 1:
        return frames[[0]].copy()
    t0, t1 = t[idx], t[idx + 1]
    a = np.clip((grid - t0) / np.maximum(t1 - t0, 1e-9), 0, 1)[:, None]
    return ((1 - a) * frames[idx] + a * frames[idx + 1]).astype(np.float32)


def mirror(frames):
    """Left-right mirror of a signer: arms and hands mirrored and swapped; the face is kept."""
    x = np.array(frames, np.float32, copy=True)
    pose = x[:, O_POSE:O_POSE + 12].reshape(-1, 6, 2)
    new = pose[:, SWAP].copy()
    new -= new[:, :1]  # re-anchor on the new left shoulder (old right shoulder)
    new[..., 0] *= -1
    x[:, O_POSE:O_POSE + 12] = new.reshape(-1, 12) * x[:, [O_PRES]]
    lh, rh = x[:, O_LH:O_LH + 42].copy(), x[:, O_RH:O_RH + 42].copy()
    lh[:, 0::2] *= -1
    rh[:, 0::2] *= -1
    x[:, O_LH:O_LH + 42], x[:, O_RH:O_RH + 42] = rh, lh
    x[:, [O_PRES + 2, O_PRES + 3]] = x[:, [O_PRES + 3, O_PRES + 2]]
    return x
```

Note on `mirror`: the pose is re-anchored by subtracting the new anchor, which is the old right shoulder. Mirroring twice restores the original exactly because the shoulder width is the same in both orientations.

- [ ] **Step 4: Run the tests to check they pass**

Run: `cd ~/sli-ai && ~/sli-work/kvenv/bin/python -m pytest training/sentences/tests/test_feats.py -q`
Expected: 7 passed. Also run `~/sli-work/venv/bin/python -c "import training.sli_kps"` from `~/sli-ai` to confirm the old scripts still import.

- [ ] **Step 5: Commit**

```bash
git add training/__init__.py training/sli_kps.py training/sentences
git commit -m "Sentences: SF1 features (Python)"   # + attribution lines
```

---

### Task 2: SF1 features in TypeScript, with parity

**Files:**
- Create: `src/recognition/sentenceFeatures.ts`, `training/sentences/export_parity.py`, `tests/unit/sentenceFeatures.test.ts`, `tests/fixtures/sentence-parity.json`
- Modify: `training/sentences/conventions.json` is read by TS through a JSON import

**Interfaces:**
- Consumes: `RawFrame` from `src/recognition/features.ts`; `feats.from_app_raw`, `feats.resample` (Task 1)
- Produces (TS):
  - `SF_DIM = 356`, `SF_FPS = 15`
  - `sentenceFrame(raw: RawFrame): Float32Array`
  - `resampleFrames(frames: Float32Array[], tMs: number[], fps = SF_FPS): Float32Array` (flat `[T' * 356]`)
  - `resampledLength(tMs: number[], fps = SF_FPS): number`

- [ ] **Step 1: Write the fixture exporter**

```python
# training/sentences/export_parity.py
"""Fixture for tests/unit/sentenceFeatures.test.ts: cached KArSL raw frames -> SF1 in Python."""
import glob, json, os, sys
import numpy as np
from training.sentences import feats as F

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
files = sorted(glob.glob(os.path.join(ROOT, "training", "cache", "h2_*.json")) or glob.glob(os.path.join(ROOT, "training", "cache", "h1_*.json")))[:4]
rng = np.random.default_rng(1)
cases = []
for i, path in enumerate(files):
    raws = json.load(open(path))[:40]
    if i == 1:  # hand lost for a stretch + face missing (Review Focus 1)
        for r in raws[10:20]:
            r["hands"] = []
            r["face"] = None
    t = np.cumsum(rng.uniform(60, 200, len(raws))) if i >= 2 else np.arange(len(raws)) * (1000 / 30)  # jittery 5-16 fps
    frames = np.stack([F.from_app_raw(r) for r in raws])
    out = F.resample(frames, t)
    cases.append({"name": os.path.basename(path), "raws": raws, "t": t.tolist(), "frames": out.round(6).tolist()})
json.dump(cases, open(os.path.join(ROOT, "tests", "fixtures", "sentence-parity.json"), "w"))
print(len(cases), "cases")
```

Run: `cd ~/sli-ai && ~/sli-work/kvenv/bin/python -m training.sentences.export_parity`
Expected: `4 cases`, and `tests/fixtures/sentence-parity.json` exists.

- [ ] **Step 2: Write the failing test**

```ts
// tests/unit/sentenceFeatures.test.ts
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import type { RawFrame } from '../../src/recognition/features';
import { SF_DIM, resampleFrames, resampledLength, sentenceFrame } from '../../src/recognition/sentenceFeatures';

interface Case { name: string; raws: RawFrame[]; t: number[]; frames: number[][] }
const cases: Case[] = JSON.parse(readFileSync(new URL('../fixtures/sentence-parity.json', import.meta.url), 'utf8'));

describe('SF1 features match Python', () => {
  it.each(cases.map((c) => [c.name, c] as const))('%s', (_n, c) => {
    const flat = resampleFrames(c.raws.map(sentenceFrame), c.t);
    expect(resampledLength(c.t)).toBe(c.frames.length);
    expect(flat.length).toBe(c.frames.length * SF_DIM);
    c.frames.forEach((row, i) => row.forEach((v, k) => expect(flat[i * SF_DIM + k]).toBeCloseTo(v, 4)));
  });

  it('empty frame is all zeros, never NaN', () => {
    const f = sentenceFrame({ pose: null, face: null, hands: [] });
    expect(f.every((v) => v === 0)).toBe(true);
  });
});
```

- [ ] **Step 3: Run it to check it fails**

Run: `npx vitest run tests/unit/sentenceFeatures.test.ts`
Expected: FAIL, cannot resolve `sentenceFeatures`.

- [ ] **Step 4: Implement**

```ts
// src/recognition/sentenceFeatures.ts
// Browser port of training/sentences/feats.py (SF1). Parity: tests/unit/sentenceFeatures.test.ts.
import faceIdx from '../../training/face_idx.json';
import conv from '../../training/sentences/conventions.json';
import type { RawFrame } from './features';

export const SF_DIM = 356;
export const SF_FPS = 15;
const FACE_IDX: readonly number[] = faceIdx.face.slice(0, 128);
const POSE_IDX = [11, 12, 13, 14, 15, 16];
const O_FACE = 12, O_LH = 268, O_RH = 310, O_PRES = 352;

type XY = ArrayLike<number>;
const dist = (a: XY, b: XY) => Math.max(Math.hypot(a[0] - b[0], a[1] - b[1]), 1e-6);

function write(out: Float32Array, o: number, pts: XY[], ax: number, ay: number, s: number) {
  for (let i = 0; i < pts.length; i++) {
    out[o + 2 * i] = (pts[i][0] - ax) / s;
    out[o + 2 * i + 1] = (pts[i][1] - ay) / s;
  }
}

function assign(raw: RawFrame): boolean[] {
  const hands = raw.hands.slice(0, 2);
  if (!raw.pose) return hands.map((h) => (h.label === 'Left') === conv.app_left_label_is_signer_left);
  const lw = raw.pose[15], rw = raw.pose[16];
  const d = (p: XY, q: XY) => Math.hypot(p[0] - q[0], p[1] - q[1]);
  if (hands.length === 2) {
    const a = hands[0].lms[0], b = hands[1].lms[0];
    return d(a, lw) + d(b, rw) <= d(a, rw) + d(b, lw) ? [true, false] : [false, true];
  }
  return hands.map((h) => d(h.lms[0], lw) <= d(h.lms[0], rw));
}

/** One frame of raw landmarks -> 356 SF1 values. Missing parts stay zero. */
export function sentenceFrame(raw: RawFrame): Float32Array {
  const out = new Float32Array(SF_DIM);
  if (raw.pose) {
    const p = raw.pose;
    const pts = POSE_IDX.map((i) => p[i]);
    write(out, 0, pts, pts[0][0], pts[0][1], dist(p[11], p[12]));
    out[O_PRES] = 1;
  }
  if (raw.face) {
    const f = raw.face;
    const pts = FACE_IDX.map((i) => f[i]);
    write(out, O_FACE, pts, pts[1][0], pts[1][1], dist(f[33], f[263]));
    out[O_PRES + 1] = 1;
  }
  const hands = raw.hands.slice(0, 2);
  assign(raw).forEach((left, k) => {
    const l = hands[k].lms;
    write(out, left ? O_LH : O_RH, l, l[0][0], l[0][1], dist(l[2], l[17]));
    out[O_PRES + (left ? 2 : 3)] = 1;
  });
  return out;
}

export function resampledLength(tMs: number[], fps = SF_FPS): number {
  if (tMs.length === 0) return 0;
  return Math.floor((tMs[tMs.length - 1] - tMs[0]) / (1000 / fps)) + 1;
}

/** Linear interpolation onto a uniform fps grid starting at tMs[0] (as feats.resample). */
export function resampleFrames(frames: Float32Array[], tMs: number[], fps = SF_FPS): Float32Array {
  const n = resampledLength(tMs, fps);
  const out = new Float32Array(n * SF_DIM);
  if (frames.length === 1) { out.set(frames[0]); return out; }
  const step = 1000 / fps;
  let j = 0;
  for (let g = 0; g < n; g++) {
    const tg = g * step;
    while (j < frames.length - 2 && tMs[j + 1] - tMs[0] <= tg) j++;
    const t0 = tMs[j] - tMs[0], t1 = tMs[j + 1] - tMs[0];
    const a = Math.min(Math.max((tg - t0) / Math.max(t1 - t0, 1e-9), 0), 1);
    const f0 = frames[j], f1 = frames[j + 1], o = g * SF_DIM;
    for (let k = 0; k < SF_DIM; k++) out[o + k] = (1 - a) * f0[k] + a * f1[k];
  }
  return out;
}
```

The `while` walk reproduces `np.searchsorted(t, grid, side="right") - 1` clipped to `[0, T-2]`. If the parity test disagrees on jittery cases, compare index choice at exact ties first.

- [ ] **Step 5: Run the tests to check they pass**

Run: `npx vitest run tests/unit/sentenceFeatures.test.ts && npx tsc --noEmit`
Expected: 5 passed; no type errors. `tsconfig` needs `resolveJsonModule`, which is already used for `face_idx.json`.

- [ ] **Step 6: Commit**

```bash
git add src/recognition/sentenceFeatures.ts training/sentences/export_parity.py tests/unit/sentenceFeatures.test.ts tests/fixtures/sentence-parity.json
git commit -m "Sentences: SF1 features (TypeScript) + parity fixture"
```

---

### Task 3: Measure KArSL and app conventions

**Files:**
- Create: `training/sentences/probe_karsl.py`
- Modify: `training/sentences/conventions.json` (measured values, `"measured": true`)

**Interfaces:**
- Consumes: `training/cache/*.json` (raw app landmarks of KArSL test videos, named `h{1,2}_<video>.json`), `ish/s00_sent0001.pose`, and 1 KArSL `.npz` downloaded with the Kaggle CLI
- Produces: `conventions.json` keys `karsl_rh_slot_is_signer_right`, `karsl_fps`, `app_left_label_is_signer_left`

- [ ] **Step 1: Fetch a sample of the published KArSL features**

```bash
cd ~/sli-work && mkdir -p kk && for s in 0001 0100 0300; do kvenv/bin/kaggle datasets download -q youssefelkilany/word-level-arabic-sign-language-extrcted-keypoints -f karsl-kps/01-test/$s.npz -p kk; done; cd kk && for z in *.zip; do unzip -o -q "$z" && rm "$z"; done; ls
```
Expected: `0001.npz 0100.npz 0300.npz`.
Also list which splits exist: `kvenv/bin/kaggle datasets files youssefelkilany/word-level-arabic-sign-language-extrcted-keypoints --page-size 1000 | awk '{print $1}' | cut -d/ -f2 | sort | uniq -c`. Record the result in the Task 3 commit message. Expected folders are like `01-train`, `01-test`, …

- [ ] **Step 2: Write the probe**

```python
# training/sentences/probe_karsl.py
"""Measures (1) which of the app's MediaPipe handedness labels is the signer's own left,
(2) which KArSL npz hand slot is the signer's right, (3) KArSL's effective frame rate."""
import glob, json, os, sys
import numpy as np
from training import sli_kps
from training.sentences import feats as F

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
KK = os.path.expanduser("~/sli-work/kk")

# (1) app label vs nearest wrist, on cached raw landmarks with two visible hands or one
agree = total = 0
for path in glob.glob(os.path.join(ROOT, "training", "cache", "h*_*.json"))[:300]:
    for r in json.load(open(path)):
        if not r["pose"]:
            continue
        for h in r["hands"]:
            w = np.array(h["lms"][0][:2])
            near_left = np.linalg.norm(w - r["pose"][15][:2]) < np.linalg.norm(w - r["pose"][16][:2])
            agree += near_left == (h["label"] == "Left"); total += 1
left_label_is_left = agree / total > 0.5
print(f"app: label 'Left' is signer-left in {agree/total:.1%} of {total} hands")

# (2) npz slot vs our own extraction of the same video (sli_kps.features puts label 'Right' in the RH slot)
match_same = match_swapped = 0
for z in glob.glob(os.path.join(KK, "*.npz")):
    npz = np.load(z)
    for key in npz.files:
        cache = glob.glob(os.path.join(ROOT, "training", "cache", f"h*_{key}.json"))
        if not cache:
            continue
        ours = np.stack([sli_kps.features(r) for r in json.load(open(cache[0]))])
        theirs = npz[key]
        n = min(len(ours), len(theirs))
        rh_o, lh_o, rh_t = ours[:n, 142:163, :2], ours[:n, 163:184, :2], theirs[:n, 142:163, :2]
        match_same += np.abs(rh_o - rh_t).mean() < np.abs(lh_o - rh_t).mean()
        match_swapped += np.abs(rh_o - rh_t).mean() >= np.abs(lh_o - rh_t).mean()
print(f"npz RH slot = our RH slot in {match_same} videos, = our LH slot in {match_swapped}")
# our RH slot holds label 'Right'; label 'Right' is signer-right iff label 'Left' is signer-left
rh_slot_is_right = (match_same >= match_swapped) == left_label_is_left

# (3) frame rate: median wrist speed (shoulder widths per frame) vs Isharah at 25 fps
from pose_format import Pose
p = Pose.read(open(os.path.expanduser("~/sli-work/ish/s00_sent0001.pose"), "rb").read())
ish = np.stack([F.from_holistic(p.body.data[i, 0], p.body.confidence[i, 0], 1000, 1000) for i in range(len(p.body.data))])
def speed(x):
    w = x[:, 8:12]  # both wrists relative to the left shoulder
    return np.median(np.abs(np.diff(w, axis=0)).sum(1))
ks = np.concatenate([np.stack([F.from_karsl184(f) for f in np.load(z)[k]]) for z in glob.glob(os.path.join(KK, "*.npz")) for k in np.load(z).files[:5]])
karsl_fps = int(round(25 * speed(ish) / max(speed(ks), 1e-9) / 5) * 5) or 15
print(f"KArSL effective fps ~ {karsl_fps}")

conv = {"karsl_rh_slot_is_signer_right": bool(rh_slot_is_right), "karsl_fps": karsl_fps,
        "app_left_label_is_signer_left": bool(left_label_is_left), "measured": True}
json.dump(conv, open(os.path.join(os.path.dirname(__file__), "conventions.json"), "w"), indent=1)
print(conv)
```

- [ ] **Step 3: Run it**

Run: `cd ~/sli-ai && ~/sli-work/kvenv/bin/python -m training.sentences.probe_karsl`
Expected:
- The label-agreement line is clearly one-sided (above 80% or below 20%). A value near 50% means labels are random. That is fine: the app uses nearest wrist whenever pose is present.
- The npz slot line is one-sided. If it is not, set `karsl_rh_slot_is_signer_right` by majority and note it in `rounds.md`. Mirror augmentation (Task 7) makes the model tolerant either way.
- An fps between 10 and 30.

The single Isharah sample is only 54 frames. If the fps estimate looks off, download 5 more `.pose` files (`s01_sent0005.pose` etc.) and average them.

- [ ] **Step 4: Re-run Task 1–2 tests (conventions changed)**

Run: `~/sli-work/kvenv/bin/python -m training.sentences.export_parity && ~/sli-work/kvenv/bin/python -m pytest training/sentences -q && npx vitest run tests/unit/sentenceFeatures.test.ts`
Expected: all pass. If `app_left_label_is_signer_left` flipped, `test_feats.py` is unaffected because it builds hands with a pose.

- [ ] **Step 5: Commit**

```bash
git add training/sentences/probe_karsl.py training/sentences/conventions.json tests/fixtures/sentence-parity.json
git commit -m "Sentences: measured KArSL/app hand and frame-rate conventions"
```

---

### Task 4: Kaggle runner

**Files:**
- Create: `training/sentences/kaggle_run.py`, `training/sentences/tests/test_kaggle_run.py`

**Interfaces:**
- Produces: CLI `python -m training.sentences.kaggle_run <name> --entry <module> [--gpu] [--datasets a/b ...] [--kernels baraasaad/x ...] [--args "..."] [--wait] [--pull DIR]`
  - Kernel slug `baraasaad/sli-sent-<name>`; the kernel runs `python -m <entry> <args>` from the bundled repo with `SLI_OUT=/kaggle/working`.
  - `bundle() -> str`: base64 tar.gz of `training/` (without `cache/`, `__pycache__`)
  - `find_input(pattern: str) -> str` (used inside kernels): first glob match under `/kaggle/input`

- [ ] **Step 1: Write the failing test**

```python
# training/sentences/tests/test_kaggle_run.py
import base64, io, json, tarfile
from training.sentences import kaggle_run as K


def test_bundle_contains_package_not_cache():
    names = tarfile.open(fileobj=io.BytesIO(base64.b64decode(K.bundle())), mode="r:gz").getnames()
    assert "training/sentences/feats.py" in names
    assert not any("/cache/" in n or "__pycache__" in n for n in names)


def test_metadata(tmp_path):
    K.write_kernel(tmp_path, "hello", "training.sentences.hello", gpu=True, datasets=["a/b"], kernels=[], args="--x 1")
    meta = json.load(open(tmp_path / "kernel-metadata.json"))
    assert meta["id"] == "baraasaad/sli-sent-hello" and meta["enable_gpu"] is True
    assert meta["dataset_sources"] == ["a/b"] and meta["is_private"] is True
    assert "training.sentences.hello" in (tmp_path / "run.py").read_text()
```

- [ ] **Step 2: Run it to check it fails**

Run: `~/sli-work/kvenv/bin/python -m pytest training/sentences/tests/test_kaggle_run.py -q`
Expected: FAIL, `No module named ... kaggle_run`.

- [ ] **Step 3: Implement**

```python
# training/sentences/kaggle_run.py
"""Runs a module of this repo as a private Kaggle kernel (GPU optional), then waits and pulls outputs."""
import argparse, base64, glob, io, json, os, subprocess, sys, tarfile, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
KAGGLE = os.path.expanduser("~/sli-work/kvenv/bin/kaggle")
USER = "baraasaad"
PIP = "pose-format numpy onnx onnxruntime"

RUN = '''import base64, io, os, subprocess, sys, tarfile
tarfile.open(fileobj=io.BytesIO(base64.b64decode(BUNDLE)), mode="r:gz").extractall("/kaggle/src")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", *"{pip}".split()], check=False)
os.environ["SLI_OUT"] = "/kaggle/working"
os.chdir("/kaggle/src")
subprocess.run(["find", "/kaggle/input", "-maxdepth", "4", "-type", "d"], check=False)
sys.exit(subprocess.run([sys.executable, "-m", "{entry}", *{args!r}.split()]).returncode)
'''


def bundle():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        def skip(ti):
            return None if ("/cache" in ti.name or "__pycache__" in ti.name or ti.name.endswith(".log")) else ti
        tar.add(os.path.join(ROOT, "training"), arcname="training", filter=skip)
    return base64.b64encode(buf.getvalue()).decode()


def write_kernel(d, name, entry, gpu, datasets, kernels, args=""):
    os.makedirs(d, exist_ok=True)
    meta = {"id": f"{USER}/sli-sent-{name}", "title": f"sli-sent-{name}", "code_file": "run.py",
            "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": gpu,
            "enable_internet": True, "dataset_sources": datasets, "kernel_sources": kernels,
            "competition_sources": []}
    json.dump(meta, open(os.path.join(d, "kernel-metadata.json"), "w"), indent=1)
    with open(os.path.join(d, "run.py"), "w") as f:
        f.write(f"BUNDLE = {bundle()!r}\n" + RUN.format(pip=PIP, entry=entry, args=args))


def find_input(pattern):
    hits = sorted(glob.glob(os.path.join("/kaggle/input", "**", pattern), recursive=True))
    if not hits:
        raise FileNotFoundError(pattern)
    return hits[0]


def status(slug):
    out = subprocess.run([KAGGLE, "kernels", "status", slug], capture_output=True, text=True).stdout
    for s in ("complete", "error", "cancel", "running", "queued"):
        if s in out.lower():
            return s
    return out.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name"); ap.add_argument("--entry", required=True)
    ap.add_argument("--gpu", action="store_true"); ap.add_argument("--datasets", nargs="*", default=[])
    ap.add_argument("--kernels", nargs="*", default=[]); ap.add_argument("--args", default="")
    ap.add_argument("--wait", action="store_true"); ap.add_argument("--pull")
    a = ap.parse_args()
    d = os.path.expanduser(f"~/sli-work/kaggle/{a.name}")
    write_kernel(d, a.name, a.entry, a.gpu, a.datasets, a.kernels, a.args)
    subprocess.run([KAGGLE, "kernels", "push", "-p", d], check=True)
    slug = f"{USER}/sli-sent-{a.name}"
    while a.wait:
        s = status(slug)
        print(time.strftime("%H:%M"), s, flush=True)
        if s in ("complete", "error", "cancel"):
            break
        time.sleep(120)
    if a.pull:
        os.makedirs(a.pull, exist_ok=True)
        subprocess.run([KAGGLE, "kernels", "output", slug, "-p", a.pull, "-o"], check=False)


if __name__ == "__main__":
    main()
```

Also create the smoke-test entry `training/sentences/hello.py`:

```python
# training/sentences/hello.py
import json, os, sys
info = {"python": sys.version}
try:
    import torch
    info.update(torch=torch.__version__, cuda=torch.cuda.is_available(),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
except ImportError:
    info["torch"] = None
from training.sentences.kaggle_run import find_input
info["ann"] = find_input("SI/train.txt")
info["pose"] = find_input("s00_sent0001.pose")
import pose_format  # noqa: F401  (proves internet pip install works)
json.dump(info, open(os.path.join(os.environ["SLI_OUT"], "hello.json"), "w"))
print(info)
```

- [ ] **Step 4: Run the tests, then the real smoke test on Kaggle**

Run: `~/sli-work/kvenv/bin/python -m pytest training/sentences/tests/test_kaggle_run.py -q`. Expected: 2 passed.

Then:
```bash
cd ~/sli-ai && setsid nohup ~/sli-work/kvenv/bin/python -m training.sentences.kaggle_run hello --entry training.sentences.hello --gpu \
  --datasets tfmohamedyahia/isharah-1000-pose abedelmortadhajemai/annotations-isharah --wait --pull ~/sli-work/kaggle/out-hello > ~/sli-work/kaggle-hello.log 2>&1 &
```
Poll with `tail -3 ~/sli-work/kaggle-hello.log` about every 2 min. Expected: `complete`, and `~/sli-work/kaggle/out-hello/hello.json` shows `"cuda": true`, a GPU name, and both paths found.

If the push fails with an internet or phone-verification error, the Kaggle account needs phone verification for GPU and internet. Ask the user to verify it at kaggle.com → Settings, then rerun.

- [ ] **Step 5: Commit**

```bash
git add training/sentences/kaggle_run.py training/sentences/hello.py training/sentences/tests/test_kaggle_run.py
git commit -m "Sentences: Kaggle kernel runner + GPU smoke test"
```

---

### Task 5: Data preparation kernel (Isharah + KArSL → SF1 shards, vocabulary)

**Files:**
- Create: `training/sentences/vocab.py`, `training/sentences/prep.py`, `training/sentences/tests/test_vocab.py`

**Interfaces:**
- Consumes: `feats.from_holistic`, `feats.from_karsl184`, `feats.resample`, `kaggle_run.find_input`
- Produces (Kaggle kernel output of `baraasaad/sli-sent-prep`, used as a `kernel_source` by later kernels):
  - `ish_feats.npz`: key = video id (e.g. `s00_sent0001`) → float16 `[T,356]` at 15 fps
  - `ish_index.json`: `[{id, signer, gloss, text, splits: {"SI": "train|dev|test", "US": "train|dev|test"}}]`
  - `karsl_feats.npz`: key = `<signer>-<split>/<sign>/<video>` → float16 `[T,356]`; `karsl_index.json`: `[{key, sign (1-based), signer, split}]`
  - `vocab.json`: `{"version": 1, "fps": 15, "blank": 0, "glosses": [...], "lookup": {"g1 g2 ...": "text"}}`. The glosses come from SI train, sorted; class id = index + 1.
- `vocab.py`: `build_vocab(rows: list[dict]) -> dict`, `encode(vocab, gloss: str) -> list[int]`, `decode_text(vocab, ids: list[int]) -> str`

- [ ] **Step 1: Write the failing test**

```python
# training/sentences/tests/test_vocab.py
from training.sentences import vocab as V

ROWS = [{"gloss": "سوال هو", "text": "من هو", "splits": {"SI": "train"}},
        {"gloss": "هو معلم لغه اشاره", "text": "هو مدرس لغه اشاره", "splits": {"SI": "train"}},
        {"gloss": "هو جديد", "text": "هو جديد", "splits": {"SI": "dev"}}]


def test_vocab_from_train_only_and_blank_zero():
    v = V.build_vocab(ROWS)
    assert v["blank"] == 0 and "جديد" not in v["glosses"]
    assert V.encode(v, "هو معلم") == [v["glosses"].index("هو") + 1, v["glosses"].index("معلم") + 1]


def test_unknown_gloss_is_dropped_and_text_lookup():
    v = V.build_vocab(ROWS)
    assert V.encode(v, "هو جديد") == [v["glosses"].index("هو") + 1]
    ids = V.encode(v, "سوال هو")
    assert V.decode_text(v, ids) == "من هو"
    assert V.decode_text(v, V.encode(v, "هو سوال")) == "هو سوال"  # no sentence -> glosses joined
```

- [ ] **Step 2: Run it to check it fails**

Run: `~/sli-work/kvenv/bin/python -m pytest training/sentences/tests/test_vocab.py -q`
Expected: FAIL, module not found.

- [ ] **Step 3: Implement `vocab.py`**

```python
# training/sentences/vocab.py
def build_vocab(rows, split=("SI", "train")):
    train = [r for r in rows if r["splits"].get(split[0]) == split[1]]
    glosses = sorted({g for r in train for g in r["gloss"].split()})
    lookup = {r["gloss"]: r["text"] for r in rows if all(g in glosses for g in r["gloss"].split())}
    return {"version": 1, "fps": 15, "blank": 0, "glosses": glosses, "lookup": lookup}


def encode(vocab, gloss):
    index = {g: i + 1 for i, g in enumerate(vocab["glosses"])}
    return [index[g] for g in gloss.split() if g in index]


def decode_text(vocab, ids):
    key = " ".join(vocab["glosses"][i - 1] for i in ids)
    return vocab["lookup"].get(key, key)
```

`lookup` also includes dev and test sentences whose glosses are all known. It only maps a recognised gloss sequence to its written form and is never used for scoring, which is on glosses. That is the spec's "exact match shows the sentence".

- [ ] **Step 4: Implement `prep.py`**

```python
# training/sentences/prep.py
"""Kaggle kernel: Isharah-1000 .pose + KArSL npz -> SF1 at 15 fps (float16) + vocab."""
import glob, json, os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from training.sentences import feats as F
from training.sentences.kaggle_run import find_input
from training.sentences.vocab import build_vocab

OUT = os.environ.get("SLI_OUT", ".")


def read_split(path):
    rows = {}
    for line in open(path, encoding="utf-8").read().splitlines()[1:]:
        vid, gloss, text = line.split("|")
        rows[vid] = (gloss.strip(), text.strip())
    return rows


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


def main():
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
            key = vid.replace("s", "", 1).replace("_sent", "_")  # s00_sent0001 -> 00_0001 (annotation id)
            if key not in splits:
                continue
            feats[vid] = arr
            index.append({"id": vid, "signer": key[:2], **splits[key]})
    print("isharah", len(files), "files,", len(index), "annotated; missing annotations:", len(splits) - len(index))
    np.savez(os.path.join(OUT, "ish_feats.npz"), **feats)
    json.dump(index, open(os.path.join(OUT, "ish_index.json"), "w"), ensure_ascii=False)
    json.dump(build_vocab(index), open(os.path.join(OUT, "vocab.json"), "w"), ensure_ascii=False)

    kroot = os.path.dirname(os.path.dirname(find_input("karsl-kps/01-test/0001.npz")))
    kfeats, kindex = {}, []
    with ProcessPoolExecutor() as ex:
        for path, out in ex.map(karsl_one, sorted(glob.glob(os.path.join(kroot, "*", "*.npz"))), chunksize=8):
            folder = os.path.basename(os.path.dirname(path))  # e.g. 01-train
            sign = int(os.path.basename(path)[:-4])
            for k, arr in out.items():
                key = f"{folder}/{sign:04d}/{k}"
                kfeats[key] = arr
                kindex.append({"key": key, "sign": sign, "signer": folder[:2], "split": folder[3:]})
    print("karsl", len(kindex), "videos")
    np.savez(os.path.join(OUT, "karsl_feats.npz"), **kfeats)
    json.dump(kindex, open(os.path.join(OUT, "karsl_index.json"), "w"))


if __name__ == "__main__":
    main()
```

The id mapping `s00_sent0001 → 00_0001` is an assumption: the annotation ids look like `00_0001` (signer_sentence). Step 5 checks it: "missing annotations" must be about 0 and `len(index)` about 15,000.

- [ ] **Step 5: Test locally on the one sample, then run on Kaggle**

Local check: `cd ~/sli-work/ish && ~/sli-work/kvenv/bin/python -c "from training.sentences.prep import ish_one; k,a=ish_one('s00_sent0001.pose'); print(k,a.shape,a.dtype, float(abs(a).max()))"` with `PYTHONPATH=~/sli-ai`.
Expected: `s00_sent0001 (33, 356) float16 <value below 50>`. 54 frames at 25 fps become 33 at 15 fps.

Kaggle (CPU is enough; no `--gpu`):
```bash
cd ~/sli-ai && setsid nohup ~/sli-work/kvenv/bin/python -m training.sentences.kaggle_run prep --entry training.sentences.prep \
  --datasets tfmohamedyahia/isharah-1000-pose abedelmortadhajemai/annotations-isharah youssefelkilany/word-level-arabic-sign-language-extrcted-keypoints \
  --wait > ~/sli-work/kaggle-prep.log 2>&1 &
```
Expected in the kernel log (`kaggle kernels output baraasaad/sli-sent-prep -p ~/sli-work/kaggle/out-prep` pulls log and files; only pull `vocab.json`, `*_index.json` and the log, since the npz files are several GB):
- `isharah 15000 files, ~15000 annotated; missing annotations: ~0`
- `karsl <tens of thousands> videos`
- vocab with about 685 glosses: `python -c "import json;print(len(json.load(open('vocab.json'))['glosses']))"`.

If the kernel runs past Kaggle's session limit, split it with `--args ish` / `--args karsl` and gate each half of `main()` on `sys.argv`.

- [ ] **Step 6: Commit**

```bash
git add training/sentences/vocab.py training/sentences/prep.py training/sentences/tests/test_vocab.py
git commit -m "Sentences: data prep kernel (Isharah + KArSL -> SF1 15 fps) and vocabulary"
```

---

### Task 6: Model, CTC decoding and WER

**Files:**
- Create: `training/sentences/ctc.py`, `training/sentences/model.py`, `training/sentences/tests/test_model.py`

**Interfaces:**
- Produces:
  - `ctc.greedy(logprobs: np.ndarray[T,C]) -> tuple[list[int], list[float]]` (ids without blanks and repeats, mean max-prob per emitted token)
  - `ctc.wer(refs: list[list[int]], hyps: list[list[int]]) -> float` (total edits / total ref tokens)
  - `model.CONFIGS = {"large": dict(d=384, layers=12, heads=6, ff=1536, conv=15), "small": dict(d=192, layers=6, heads=4, ff=768, conv=15)}`
  - `model.SignCTC(n_classes: int, d, layers, heads, ff, conv, dropout=0.1)`; `forward(x: [B,T,356], lengths: [B]) -> (logprobs [B,T',C], out_lengths [B])` with `T' = ceil(T/2)`
  - `model.build(name: str, n_classes: int) -> SignCTC`

- [ ] **Step 1: Write the failing tests**

```python
# training/sentences/tests/test_model.py
import numpy as np
import pytest
import torch
from training.sentences import ctc, model as M


def test_greedy_collapses_repeats_and_blanks():
    lp = np.log(np.eye(4)[[0, 2, 2, 0, 2, 3, 3, 0]] * 0.97 + 0.01)
    ids, conf = ctc.greedy(lp)
    assert ids == [2, 2, 3] and len(conf) == 3


def test_wer():
    assert ctc.wer([[1, 2, 3]], [[1, 2, 3]]) == 0
    assert ctc.wer([[1, 2, 3, 4]], [[1, 3, 4, 5]]) == pytest.approx(0.5)


@pytest.mark.parametrize("name", ["small", "large"])
def test_shapes_and_padding_invariance(name):
    torch.manual_seed(0)
    m = M.build(name, 10).eval()
    x = torch.randn(2, 41, 356)
    lengths = torch.tensor([41, 20])
    lp, ol = m(x, lengths)
    assert lp.shape == (2, 21, 10) and ol.tolist() == [21, 10]
    alone, _ = m(x[1:, :20], torch.tensor([20]))
    assert torch.allclose(lp[1, :10], alone[0], atol=1e-4)  # padding must not leak


def test_small_is_under_20mb():
    n = sum(p.numel() for p in M.build("small", 700).parameters())
    assert n * 4 < 20e6
```

- [ ] **Step 2: Run the tests to check they fail**

Install once: `~/.local/bin/uv pip install -q -p ~/sli-work/kvenv torch --index-url https://download.pytorch.org/whl/cpu && ~/.local/bin/uv pip install -q -p ~/sli-work/kvenv onnx onnxruntime`
Run: `~/sli-work/kvenv/bin/python -m pytest training/sentences/tests/test_model.py -q`
Expected: FAIL, modules missing.

- [ ] **Step 3: Implement**

```python
# training/sentences/ctc.py
import numpy as np


def greedy(logprobs):
    best = logprobs.argmax(-1)
    probs = np.exp(logprobs.max(-1))
    ids, conf, prev, run = [], [], 0, []
    for k, p in zip(best, probs):
        if k != prev and prev != 0:
            conf.append(float(np.mean(run)))
        if k != 0 and k != prev:
            ids.append(int(k)); run = []
        if k != 0:
            run.append(p)
        prev = k
    if prev != 0:
        conf.append(float(np.mean(run)))
    return ids, conf


def _edits(r, h):
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
    return d[len(h)]


def wer(refs, hyps):
    return sum(_edits(r, h) for r, h in zip(refs, hyps)) / max(sum(len(r) for r in refs), 1)
```

```python
# training/sentences/model.py
"""Conformer-lite encoder + CTC. Input SF1 [B,T,356] at 15 fps, output log-probs at 7.5 fps."""
import math
import torch
from torch import nn

CONFIGS = {"large": dict(d=384, layers=12, heads=6, ff=1536, conv=15),
           "small": dict(d=192, layers=6, heads=4, ff=768, conv=15)}


class ConvModule(nn.Module):
    def __init__(self, d, k, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.pw1 = nn.Linear(d, 2 * d)
        self.dw = nn.Conv1d(d, d, k, padding=k // 2, groups=d)
        self.bn = nn.BatchNorm1d(d)
        self.pw2 = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):  # mask: [B,T] True = padding
        y = nn.functional.glu(self.pw1(self.norm(x)), dim=-1)
        y = y.masked_fill(mask[..., None], 0).transpose(1, 2)
        y = nn.functional.silu(self.bn(self.dw(y))).transpose(1, 2)
        return x + self.drop(self.pw2(y))


class Block(nn.Module):
    def __init__(self, d, heads, ff, conv, dropout):
        super().__init__()
        self.ff1 = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, ff), nn.SiLU(), nn.Dropout(dropout), nn.Linear(ff, d))
        self.norm_att = nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.conv = ConvModule(d, conv, dropout)
        self.ff2 = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, ff), nn.SiLU(), nn.Dropout(dropout), nn.Linear(ff, d))
        self.norm_out = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = x + 0.5 * self.drop(self.ff1(x))
        a = self.norm_att(x)
        x = x + self.drop(self.att(a, a, a, key_padding_mask=mask, need_weights=False)[0])
        x = self.conv(x, mask)
        x = x + 0.5 * self.drop(self.ff2(x))
        return self.norm_out(x)


class SignCTC(nn.Module):
    def __init__(self, n_classes, d, layers, heads, ff, conv, dropout=0.1):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(356, d), nn.LayerNorm(d), nn.SiLU())
        self.sub = nn.Conv1d(d, d, 3, stride=2, padding=1)  # 15 fps -> 7.5 fps
        self.pos = nn.Parameter(torch.zeros(1, 4096, d))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(Block(d, heads, ff, conv, dropout) for _ in range(layers))
        self.head = nn.Linear(d, n_classes)

    def forward(self, x, lengths):
        valid = torch.arange(x.shape[1], device=x.device)[None] < lengths[:, None]
        h = self.inp(x) * valid[..., None]
        h = self.sub(h.transpose(1, 2)).transpose(1, 2)
        out_len = (lengths + 1) // 2
        mask = torch.arange(h.shape[1], device=x.device)[None] >= out_len[:, None]
        h = h + self.pos[:, : h.shape[1]]
        for b in self.blocks:
            h = b(h, mask)
        return self.head(h).log_softmax(-1), out_len


def build(name, n_classes):
    return SignCTC(n_classes, **CONFIGS[name])
```

The positional table caps inputs at 8192 frames (9 minutes), far above the 450-frame API cap. BatchNorm in eval mode uses running statistics, so padding cannot leak across samples at inference. The padding test runs in `eval()`.

- [ ] **Step 4: Run the tests to check they pass**

Run: `~/sli-work/kvenv/bin/python -m pytest training/sentences/tests/test_model.py -q`
Expected: 5 passed. If `test_shapes_and_padding_invariance` fails at `atol=1e-4`, first check that the depthwise conv sees zeros past the valid length. The `masked_fill` happens before `dw`, and `sub` receives `h * valid`.

- [ ] **Step 5: Commit**

```bash
git add training/sentences/ctc.py training/sentences/model.py training/sentences/tests/test_model.py
git commit -m "Sentences: Conformer-lite CTC model, greedy decoding, WER"
```

---

### Task 7: Dataset and augmentation

**Files:**
- Create: `training/sentences/data.py`, `training/sentences/tests/test_data.py`

**Interfaces:**
- Consumes: `feats.mirror`, `feats.O_*`, `vocab.encode`
- Produces:
  - `augment(x: np.ndarray[T,356], rng: np.random.Generator) -> np.ndarray[T',356]`
  - `class SeqDataset(torch.utils.data.Dataset)`: `__init__(self, feats: dict[str,np.ndarray], items: list[tuple[str, list[int]]], train: bool, seed=0)`; `__getitem__ -> (x: float32 tensor [T,356], y: long tensor [L])`
  - `collate(batch) -> (x [B,Tmax,356], x_len [B], y [sum L], y_len [B])`
  - `bucket_batches(lengths: list[int], max_frames: int, rng) -> list[list[int]]` (batches of similar length, total frames ≤ max_frames)

- [ ] **Step 1: Write the failing tests**

```python
# training/sentences/tests/test_data.py
import numpy as np
import torch
from training.sentences import data as D, feats as F

rng = np.random.default_rng(0)


def sample(T=60):
    x = rng.normal(0, 0.3, (T, F.FEAT_DIM)).astype(np.float32)
    x[:, F.O_PRES:] = 1
    return x


def test_augment_keeps_layout_and_presence_semantics():
    for _ in range(50):
        y = D.augment(sample(), rng)
        assert y.shape[1] == F.FEAT_DIM and 30 <= y.shape[0] <= 100
        assert np.isfinite(y).all()
        for part, (o, n) in {2: (F.O_LH, 42), 3: (F.O_RH, 42)}.items():
            gone = y[:, F.O_PRES + part] == 0
            assert not y[gone, o:o + n].any()  # a dropped hand is all zeros, flag 0


def test_collate_and_buckets():
    ds = D.SeqDataset({"a": sample(40), "b": sample(25)}, [("a", [1, 2]), ("b", [3])], train=False)
    x, xl, y, yl = D.collate([ds[0], ds[1]])
    assert x.shape == (2, 40, 356) and xl.tolist() == [40, 25] and y.tolist() == [1, 2, 3] and yl.tolist() == [2, 1]
    b = D.bucket_batches([10, 300, 20, 310, 15], max_frames=640, rng=rng)
    assert sorted(i for bb in b for i in bb) == [0, 1, 2, 3, 4]
    assert all(max([10, 300, 20, 310, 15][i] for i in bb) * len(bb) <= 640 for bb in b)
```

- [ ] **Step 2: Run the tests to check they fail**

Run: `~/sli-work/kvenv/bin/python -m pytest training/sentences/tests/test_data.py -q`
Expected: FAIL, module missing.

- [ ] **Step 3: Implement**

```python
# training/sentences/data.py
"""Training data: SF1 sequences + augmentation aimed at unseen signers and cameras."""
import numpy as np
import torch
from training.sentences import feats as F

PARTS = [(F.O_POSE, 6, 0), (F.O_FACE, 128, 1), (F.O_LH, 21, 2), (F.O_RH, 21, 3)]


def _affine(pts, rng, rot, scale):
    a = np.deg2rad(rng.uniform(-rot, rot))
    sx, sy = rng.uniform(1 - scale, 1 + scale, 2)
    m = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) @ np.diag([sx, sy])
    return pts @ m.T


def augment(x, rng):
    x = x.astype(np.float32, copy=True)
    if rng.random() < 0.5:
        x = F.mirror(x)
    T = len(x)
    for o, n, k in PARTS:  # per-part shape/aspect/rotation (body proportions, camera angle)
        pts = x[:, o:o + 2 * n].reshape(T, n, 2)
        rot, scale = (15, 0.2) if k == 0 else (10, 0.15)
        pts = _affine(pts, rng, rot, scale) + rng.normal(0, 0.01, pts.shape)
        x[:, o:o + 2 * n] = pts.reshape(T, -1) * x[:, [F.O_PRES + k]]
    speed = rng.uniform(0.7, 1.3)  # signing speed
    t = np.arange(T) * speed
    x = F.resample(x, t * 1000.0 / F.FPS)
    T = len(x)
    for o, n, k in PARTS[2:]:  # short hand dropouts (tracking loss)
        if rng.random() < 0.3:
            s = rng.integers(0, T); e = min(T, s + rng.integers(1, 8))
            x[s:e, o:o + 2 * n] = 0; x[s:e, F.O_PRES + k] = 0
    if rng.random() < 0.05:
        x[:, F.O_FACE:F.O_FACE + 256] = 0; x[:, F.O_PRES + 1] = 0
    return x


class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, feats, items, train, seed=0):
        self.feats, self.items, self.train = feats, items, train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        key, ids = self.items[i]
        x = np.asarray(self.feats[key], np.float32)
        if self.train:
            x = augment(x, self.rng)
        return torch.from_numpy(x), torch.tensor(ids, dtype=torch.long)


def collate(batch):
    xs, ys = zip(*batch)
    xl = torch.tensor([len(x) for x in xs])
    x = torch.zeros(len(xs), int(xl.max()), F.FEAT_DIM)
    for i, v in enumerate(xs):
        x[i, : len(v)] = v
    return x, xl, torch.cat(ys), torch.tensor([len(y) for y in ys])


def bucket_batches(lengths, max_frames, rng):
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches, cur = [], []
    for i in order:
        if cur and max(lengths[j] for j in cur + [i]) * (len(cur) + 1) > max_frames:
            batches.append(cur); cur = []
        cur.append(i)
    if cur:
        batches.append(cur)
    rng.shuffle(batches)
    return batches
```

Mirroring before the per-part affine is deliberate: `mirror()` expects the SF1 anchors unchanged. `F.resample` of an already-resampled grid with a stretched time axis gives the speed change. The hand dropout test holds because the affine multiplies by the presence flag and the dropout zeroes both.

- [ ] **Step 4: Run the tests to check they pass**

Run: `~/sli-work/kvenv/bin/python -m pytest training/sentences/tests/test_data.py -q`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add training/sentences/data.py training/sentences/tests/test_data.py
git commit -m "Sentences: dataset, bucketing and signer/camera augmentation"
```

---

### Task 8: Training entry point + first real round (large, KArSL pre-train → Isharah)

**Files:**
- Create: `training/sentences/train.py`, `training/sentences/tests/test_train.py`, `training/sentences/rounds.md`

**Interfaces:**
- Consumes: `data.*`, `model.build`, `ctc.greedy/wer`, `vocab.encode`, prep outputs (`find_input("ish_feats.npz")` etc.)
- Produces:
  - CLI `python -m training.sentences.train --stage {karsl,isharah,distill} --model {large,small} [--init ckpt.pt] [--teacher ckpt.pt] [--epochs N] [--protocol SI|US] [--max-frames 12000] [--lr 1e-3] [--limit N] [--data DIR]`
  - Outputs to `$SLI_OUT`: `<stage>-<model>.pt` = `{"state": state_dict, "config": name, "n_classes": C, "dev_wer": float, "epoch": int}` (best dev), `<stage>-<model>.log.json` (per-epoch train loss, dev WER)
  - `evaluate_split(model, feats, items, device) -> float` (WER, greedy)

- [ ] **Step 1: Write the failing overfit test**

```python
# training/sentences/tests/test_train.py
import numpy as np
import torch
from training.sentences import train as T, model as M, data as D, feats as F


def test_tiny_model_overfits_four_sentences():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    feats, items = {}, []
    protos = {g: rng.normal(0, 1, (6, F.FEAT_DIM)).astype(np.float32) for g in (1, 2, 3)}
    for k, seq in enumerate([[1, 2], [2, 3], [3, 1, 2], [1]]):
        feats[str(k)] = np.concatenate([protos[g] for g in seq])
        items.append((str(k), seq))
    m = M.SignCTC(4, d=64, layers=2, heads=4, ff=128, conv=3, dropout=0.0)
    T.fit(m, feats, items, feats, items, epochs=150, lr=3e-3, max_frames=200, device="cpu", augment=False, log=None)
    assert T.evaluate_split(m, feats, items, "cpu") == 0.0
```

- [ ] **Step 2: Run it to check it fails**

Run: `~/sli-work/kvenv/bin/python -m pytest training/sentences/tests/test_train.py -q`
Expected: FAIL, `training.sentences.train` missing.

- [ ] **Step 3: Implement**

```python
# training/sentences/train.py
"""Stages: karsl (isolated-sign pre-training), isharah (sentences), distill (small from large)."""
import argparse, json, math, os, time
import numpy as np
import torch
from torch import nn
from training.sentences import ctc, data as D, model as M
from training.sentences.vocab import encode

OUT = os.environ.get("SLI_OUT", ".")


def evaluate_split(model, feats, items, device):
    model.eval()
    refs, hyps = [], []
    with torch.no_grad():
        for key, ids in items:
            x = torch.from_numpy(np.asarray(feats[key], np.float32))[None].to(device)
            lp, _ = model(x, torch.tensor([x.shape[1]], device=device))
            hyps.append(ctc.greedy(lp[0].float().cpu().numpy())[0]); refs.append(ids)
    return ctc.wer(refs, hyps)


def fit(model, feats, items, dev_feats, dev_items, epochs, lr, max_frames, device, augment=True,
        log=None, teacher=None, alpha=1.0, ckpt=None, meta=None):
    model.to(device)
    ds = D.SeqDataset(feats, items, train=augment)
    lengths = [len(feats[k]) for k, _ in items]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    rng = np.random.default_rng(0)
    steps = epochs * len(D.bucket_batches(lengths, max_frames, rng))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(steps, 1), pct_start=0.1)
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    best, history = math.inf, []
    for ep in range(epochs):
        model.train(); total = 0.0
        for batch in D.bucket_batches(lengths, max_frames, rng):
            x, xl, y, yl = D.collate([ds[i] for i in batch])
            x, xl = x.to(device), xl.to(device)
            with torch.autocast(device, enabled=device == "cuda"):
                lp, ol = model(x, xl)
            loss = ctc_loss(lp.float().transpose(0, 1), y, ol.cpu(), yl)
            if teacher is not None:  # frame-level distillation, same stride and fps
                with torch.no_grad():
                    tlp, _ = teacher(x, xl)
                valid = (torch.arange(lp.shape[1], device=device)[None] < ol[:, None]).float()
                kl = (tlp.float().exp() * (tlp.float() - lp.float())).sum(-1)
                loss = loss + alpha * (kl * valid).sum() / valid.sum()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            total += float(loss)
        dev = evaluate_split(model, dev_feats, dev_items, device)
        history.append({"epoch": ep, "loss": total, "dev_wer": dev, "t": time.time()})
        print(json.dumps(history[-1]), flush=True)
        if dev < best:
            best = dev
            if ckpt:
                torch.save({"state": model.state_dict(), **(meta or {}), "dev_wer": dev, "epoch": ep}, ckpt)
        if log:
            json.dump(history, open(log, "w"))
    return best


def load_data(data_dir=None):
    from training.sentences.kaggle_run import find_input
    path = (lambda n: os.path.join(data_dir, n)) if data_dir else find_input
    ish = np.load(path("ish_feats.npz"))
    return (ish, json.load(open(path("ish_index.json"), encoding="utf-8")),
            json.load(open(path("vocab.json"), encoding="utf-8")),
            lambda: (np.load(path("karsl_feats.npz")), json.load(open(path("karsl_index.json")))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["karsl", "isharah", "distill"])
    ap.add_argument("--model", default="large"); ap.add_argument("--init"); ap.add_argument("--teacher")
    ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-frames", type=int, default=12000); ap.add_argument("--protocol", default="SI")
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--data")
    a = ap.parse_args()
    dev_ = "cuda" if torch.cuda.is_available() else "cpu"
    ish, index, vocab, karsl = load_data(a.data)
    name = f"{a.stage}-{a.model}" + ("" if a.protocol == "SI" else f"-{a.protocol}")
    ckpt, log = os.path.join(OUT, name + ".pt"), os.path.join(OUT, name + ".log.json")

    def split(part):
        rows = [r for r in index if r["splits"].get(a.protocol) == part]
        return [(r["id"], encode(vocab, r["gloss"])) for r in rows][: a.limit or None]

    if a.stage == "karsl":
        kf, kidx = karsl()
        m = M.build(a.model, 503)  # 502 signs + blank
        tr = [(r["key"], [r["sign"]]) for r in kidx if r["split"] == "train"][: a.limit or None]
        dv = [(r["key"], [r["sign"]]) for r in kidx if r["split"] == "test"][: 2000]
        fit(m, kf, tr, kf, dv, a.epochs, a.lr, a.max_frames, dev_, log=log, ckpt=ckpt,
            meta={"config": a.model, "n_classes": 503})
        return
    C = len(vocab["glosses"]) + 1
    m = M.build(a.model, C)
    if a.init:
        state = torch.load(a.init, map_location="cpu")["state"]
        state = {k: v for k, v in state.items() if not k.startswith("head.")}
        print("init:", m.load_state_dict(state, strict=False))
    teacher = None
    if a.stage == "distill":
        t = torch.load(a.teacher, map_location="cpu")
        teacher = M.build(t["config"], C); teacher.load_state_dict(t["state"]); teacher.to(dev_).eval()
    fit(m, ish, split("train"), ish, split("dev"), a.epochs, a.lr, a.max_frames, dev_, log=log, ckpt=ckpt,
        teacher=teacher, meta={"config": a.model, "n_classes": C, "protocol": a.protocol})


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to check they pass**

Run: `~/sli-work/kvenv/bin/python -m pytest training/sentences -q`
Expected: all tests pass, including the overfit test (WER 0.0 within 150 epochs on CPU, under 1 min). If it does not reach 0, raise epochs to 300 before changing code; the test is only about whether learning works.

- [ ] **Step 5: Get the paper's baseline numbers**

Fetch the Isharah paper (arXiv version of "Isharah: A Large-Scale Multi-Scene Dataset for Continuous Sign Language Recognition", Alyami et al.; also https://snalyami.github.io/Isharah_CSLR/). Record in `rounds.md` its SI and US WER for the best pose- or keypoint-based baseline, and for the best baseline of any kind. That gives the stopping target from the spec.

```markdown
# Sentence model rounds

Target (Isharah paper, SI split, dev/test WER): pose-based <x/y> (<method>), best overall <x/y> (<method>). Source: <url>, table <n>.

| Round | Date | Model | Change (one thing) | Epochs | Dev WER | Kaggle GPU h | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
```

Fill `<...>` with the real values from the paper; they are data to look up, not to guess. If the paper cannot be retrieved, ask the user before continuing.

- [ ] **Step 6: Round 1 on Kaggle (GPU)**

```bash
cd ~/sli-ai && setsid nohup sh -c '
K="~/sli-work/kvenv/bin/python -m training.sentences.kaggle_run"
eval $K r1-karsl --entry training.sentences.train --gpu --kernels baraasaad/sli-sent-prep --args "--stage karsl --model large --epochs 30" --wait --pull ~/sli-work/kaggle/out-r1-karsl &&
eval $K r1-ish --entry training.sentences.train --gpu --kernels baraasaad/sli-sent-prep baraasaad/sli-sent-r1-karsl --args "--stage isharah --model large --epochs 60 --init /kaggle/input/sli-sent-r1-karsl/karsl-large.pt" --wait --pull ~/sli-work/kaggle/out-r1-ish
' > ~/sli-work/kaggle-r1.log 2>&1 &
```
The mount path of a kernel source is printed by the `find` in every kernel's log. If `/kaggle/input/sli-sent-r1-karsl/` differs, fix `--init` accordingly.
Poll `tail -5 ~/sli-work/kaggle-r1.log` roughly every 10 min. Expected:
- `karsl-large.log.json` dev WER (= isolated-sign error rate) below 0.2 by the end.
- `isharah-large.log.json` dev WER decreasing over the epochs.

Record the row in `rounds.md`, with the GPU hours from the kernel log timestamps.

If an epoch takes more than about 5 min, halve `--max-frames` or use `--epochs 40`. Kaggle's GPU session limit is about 12 h.

- [ ] **Step 7: Commit**

```bash
git add training/sentences/train.py training/sentences/tests/test_train.py training/sentences/rounds.md
git commit -m "Sentences: training (KArSL pre-train, Isharah CTC, distillation) + round 1"
```

---

### Task 9: Improvement rounds until the stopping rule

**Files:**
- Modify: `training/sentences/rounds.md`, plus whichever file a round changes (`model.py` CONFIGS, `data.py` augmentation strengths, `train.py` defaults)

**Interfaces:**
- Consumes: Task 8 CLI
- Produces: the best large checkpoint `~/sli-work/kaggle/best/isharah-large.pt` (and the round that made it recorded in `rounds.md`)

The protocol is fixed; the content of each round is chosen from the evidence of the previous one.

- [ ] **Step 1: One change per round, compared on SI dev WER only.** Candidates, in the order to try them:
  1. No KArSL pre-train (`--init` omitted), to learn whether pre-training helps.
  2. Epochs 60 → 100, if the dev WER was still falling at the end.
  3. Augmentation strength: rotation 15 → 25, scale 0.2 → 0.3, speed 0.6–1.4.
  4. Model width/depth: `large` d=512/16 layers (add as `CONFIGS["xl"]`) if train loss is far below dev and augmentation helped; or dropout 0.1 → 0.2 if it overfits.
  5. Learning rate 1e-3 → 5e-4 or 2e-3.
  Each round: edit, run the tests (`pytest training/sentences -q`), commit, push the kernel with a new name `rN-ish`, record the row.
- [ ] **Step 2: Stop** when dev WER ≤ the paper's pose-based SI dev WER (or its best overall if it has no pose baseline), **or** when two consecutive rounds do not improve dev WER by at least 0.5 points absolute. Write the reason in `rounds.md`.
- [ ] **Step 3: Budget guard.** Check the weekly GPU quota (`kaggle kernels list --mine` shows runs; the quota is on the Kaggle site). If fewer than 6 GPU h remain for the week, pause and tell the user rather than start a run that will be cut off.
- [ ] **Step 4: Tell the user the numbers after every round.** One line: round, change, dev WER, versus the target.
- [ ] **Step 5: Commit** `rounds.md` (and the winning config) with message `Sentences: rounds 2..N (<stop reason>)`.

---

### Task 10: Distil the phone model

**Files:**
- Modify: `training/sentences/rounds.md`

**Interfaces:**
- Consumes: best `isharah-large.pt` (Task 9); `train.py --stage distill`
- Produces: `distill-small.pt` (best dev)

- [ ] **Step 1: Baseline small without a teacher (for the model card's comparison)**

```bash
~/sli-work/kvenv/bin/python -m training.sentences.kaggle_run s1-plain --entry training.sentences.train --gpu --kernels baraasaad/sli-sent-prep baraasaad/sli-sent-r1-karsl --args "--stage isharah --model small --epochs 60" --wait --pull ~/sli-work/kaggle/out-s1-plain
```

- [ ] **Step 2: Distilled small.** Upload the best large checkpoint as a private dataset first, so kernels can read it:

```bash
mkdir -p ~/sli-work/kaggle/ds-large && cp ~/sli-work/kaggle/best/isharah-large.pt ~/sli-work/kaggle/ds-large/ &&
printf '{"title":"sli-sent-large","id":"baraasaad/sli-sent-large","licenses":[{"name":"CC-BY-NC-SA-4.0"}]}' > ~/sli-work/kaggle/ds-large/dataset-metadata.json &&
~/sli-work/kvenv/bin/kaggle datasets create -p ~/sli-work/kaggle/ds-large   # later versions: kaggle datasets version -p ... -m "roundN"
~/sli-work/kvenv/bin/python -m training.sentences.kaggle_run s1-distill --entry training.sentences.train --gpu --datasets baraasaad/sli-sent-large --kernels baraasaad/sli-sent-prep --args "--stage distill --model small --teacher /kaggle/input/sli-sent-large/isharah-large.pt --epochs 80" --wait --pull ~/sli-work/kaggle/out-s1-distill
```
The dataset is private by default. Expected: distilled small dev WER lower than plain small. If it is not, try `alpha` 0.5 (add a `--alpha` flag to `train.py` passing into `fit`) as one more round. Record both rows.

- [ ] **Step 3: Commit** `rounds.md`: `Sentences: distilled phone model (dev WER x vs plain y)`.

---

### Task 11: Export to ONNX (large fp32, small int8) + parity + latency

**Files:**
- Create: `training/sentences/export.py`, `training/sentences/tests/test_export.py`
- Create (artefacts): `public/models/sentences-small.onnx`, `public/models/sentences-vocab.json`, `~/sli-work/models/sentences-large.onnx`
- Create: `tests/fixtures/sentence-model-parity.json`

**Interfaces:**
- Consumes: checkpoints from Tasks 9–10, `vocab.json` from prep
- Produces: ONNX with input `feats` float32 `[1, T, 356]` (dynamic T) and output `logprobs` float32 `[1, ceil(T/2), C]`; `export.export(ckpt: str, out: str, quantize: bool) -> None`; `export.latency(onnx_path: str, frames=300, threads=1) -> float` (median seconds over 5 runs)

- [ ] **Step 1: Write the failing test**

```python
# training/sentences/tests/test_export.py
import numpy as np, onnxruntime as ort, torch
from training.sentences import export as E, model as M


def test_onnx_matches_torch_dynamic_length(tmp_path):
    torch.manual_seed(0)
    m = M.SignCTC(12, d=64, layers=2, heads=4, ff=128, conv=3).eval()
    ck = tmp_path / "m.pt"
    torch.save({"state": m.state_dict(), "config": None, "n_classes": 12,
                "arch": dict(d=64, layers=2, heads=4, ff=128, conv=3)}, ck)
    out = tmp_path / "m.onnx"
    E.export(str(ck), str(out), quantize=False)
    s = ort.InferenceSession(str(out))
    for T in (9, 57):
        x = np.random.default_rng(T).normal(size=(1, T, 356)).astype(np.float32)
        ref, _ = m(torch.from_numpy(x), torch.tensor([T]))
        got = s.run(["logprobs"], {"feats": x})[0]
        assert got.shape == (1, (T + 1) // 2, 12)
        assert np.allclose(got, ref.detach().numpy(), atol=1e-4)
```

- [ ] **Step 2: Run it to check it fails**

Run: `~/sli-work/kvenv/bin/python -m pytest training/sentences/tests/test_export.py -q`
Expected: FAIL, module missing.

- [ ] **Step 3: Implement**

```python
# training/sentences/export.py
import argparse, json, os, statistics, time
import numpy as np, onnxruntime as ort, torch
from training.sentences import model as M


class Wrapped(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, feats):
        lp, _ = self.m(feats, torch.full((1,), feats.shape[1], dtype=torch.long))
        return lp


def export(ckpt, out, quantize):
    c = torch.load(ckpt, map_location="cpu")
    arch = c.get("arch") or M.CONFIGS[c["config"]]
    m = M.SignCTC(c["n_classes"], **arch); m.load_state_dict(c["state"]); m.eval()
    x = torch.zeros(1, 60, 356)
    tmp = out + ".fp32.onnx" if quantize else out
    torch.onnx.export(Wrapped(m), (x,), tmp, input_names=["feats"], output_names=["logprobs"],
                      dynamic_axes={"feats": {1: "T"}, "logprobs": {1: "T2"}}, opset_version=17, dynamo=False)
    if quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        quantize_dynamic(tmp, out, weight_type=QuantType.QInt8)
        os.remove(tmp)


def latency(path, frames=300, threads=1):
    o = ort.SessionOptions(); o.intra_op_num_threads = threads; o.inter_op_num_threads = 1
    s = ort.InferenceSession(path, o)
    x = np.random.default_rng(0).normal(size=(1, frames, 356)).astype(np.float32)
    s.run(None, {"feats": x})
    times = []
    for _ in range(5):
        t = time.perf_counter(); s.run(None, {"feats": x}); times.append(time.perf_counter() - t)
    return statistics.median(times)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("out"); ap.add_argument("--quantize", action="store_true")
    ap.add_argument("--latency", action="store_true")
    a = ap.parse_args()
    export(a.ckpt, a.out, a.quantize)
    print(a.out, f"{os.path.getsize(a.out) / 1e6:.1f} MB")
    if a.latency:
        print(f"300 frames, 1 thread: {latency(a.out):.2f} s")
```

- [ ] **Step 4: Run the test, then export the real models and check the limits**

```bash
cd ~/sli-ai && ~/sli-work/kvenv/bin/python -m pytest training/sentences/tests/test_export.py -q
nice -n 19 ~/sli-work/kvenv/bin/python -m training.sentences.export ~/sli-work/kaggle/best/isharah-large.pt ~/sli-work/models/sentences-large.onnx --latency
nice -n 19 ~/sli-work/kvenv/bin/python -m training.sentences.export ~/sli-work/kaggle/out-s1-distill/distill-small.pt public/models/sentences-small.onnx --quantize --latency
cp ~/sli-work/kaggle/out-prep/vocab.json public/models/sentences-vocab.json
```
Expected:
- the test passes;
- large: `300 frames, 1 thread: ≤ 2.00 s`. If it is slower, go back to Task 9 and pick the best dev-WER config that meets 2 s. A slower model is not released, per the spec.
- small: `≤ 20.0 MB`.

Then confirm int8 did not hurt. Run dev WER of the quantised small model with `evaluate.py` (Task 12, dev split only) against the fp32 dev WER from its checkpoint. If quantisation costs more than 1 point, ship fp32 small when it is ≤ 20 MB, and record the choice in `rounds.md`.

- [ ] **Step 5: Model parity fixture for the browser**

Extend `export_parity.py` with a `--model` mode: for the 4 parity cases, run `public/models/sentences-small.onnx` on the Python SF1 frames and store `{"name", "logprobs_head": first 20 values, "ids": greedy ids}` in `tests/fixtures/sentence-model-parity.json`:

```python
# appended to training/sentences/export_parity.py
if "--model" in sys.argv:
    import onnxruntime as ort
    from training.sentences import ctc
    s = ort.InferenceSession(os.path.join(ROOT, "public", "models", "sentences-small.onnx"))
    res = []
    for c in cases:
        lp = s.run(["logprobs"], {"feats": np.array(c["frames"], np.float32)[None]})[0][0]
        res.append({"name": c["name"], "logprobs_head": lp.ravel()[:20].round(5).tolist(), "ids": ctc.greedy(lp)[0]})
    json.dump(res, open(os.path.join(ROOT, "tests", "fixtures", "sentence-model-parity.json"), "w"))
```

Run: `~/sli-work/kvenv/bin/python -m training.sentences.export_parity --model`

- [ ] **Step 6: Commit**

```bash
git add training/sentences/export.py training/sentences/tests/test_export.py training/sentences/export_parity.py public/models/sentences-small.onnx public/models/sentences-vocab.json tests/fixtures/sentence-model-parity.json training/sentences/rounds.md
git commit -m "Sentences: ONNX export (large fp32, small int8), latency and size checks"
```
The large ONNX is not committed (it can be big); it is published on Hugging Face in Task 17 and copied into the server image in Task 13.

---

### Task 12: Final evaluation (SI test once, US, ArabSign)

**Files:**
- Create: `training/sentences/evaluate.py`, `training/sentences/arabsign.py`
- Modify: `src/data/metrics.json` (add a `sentences` key), `training/sentences/rounds.md`

**Interfaces:**
- Consumes: ONNX models (Task 11), prep outputs, `ctc.greedy/wer`, `vocab.encode`
- Produces: CLI `python -m training.sentences.evaluate <onnx> --protocol SI|US --part dev|test [--data DIR]` → prints and writes `{"model", "protocol", "part", "wer", "n", "sentence_acc"}` to `$SLI_OUT/eval-<model>-<protocol>-<part>.json`

- [ ] **Step 1: Implement `evaluate.py`**

```python
# training/sentences/evaluate.py
import argparse, json, os
import numpy as np, onnxruntime as ort
from training.sentences import ctc
from training.sentences.train import load_data
from training.sentences.vocab import encode

ap = argparse.ArgumentParser()
ap.add_argument("onnx"); ap.add_argument("--protocol", default="SI"); ap.add_argument("--part", default="dev")
ap.add_argument("--data")
a = ap.parse_args()
ish, index, vocab, _ = load_data(a.data)
s = ort.InferenceSession(a.onnx)
refs, hyps = [], []
for r in index:
    if r["splits"].get(a.protocol) != a.part:
        continue
    lp = s.run(["logprobs"], {"feats": np.asarray(ish[r["id"]], np.float32)[None]})[0][0]
    hyps.append(ctc.greedy(lp)[0]); refs.append(encode(vocab, r["gloss"]))
res = {"model": os.path.basename(a.onnx), "protocol": a.protocol, "part": a.part, "n": len(refs),
       "wer": round(ctc.wer(refs, hyps), 4), "sentence_acc": round(float(np.mean([h == r for h, r in zip(hyps, refs)])), 4)}
print(res)
json.dump(res, open(os.path.join(os.environ.get("SLI_OUT", "."), f"eval-{res['model']}-{a.protocol}-{a.part}.json"), "w"))
```

Run it on Kaggle (CPU), because `ish_feats.npz` lives there: upload both ONNX files as a new version of the `baraasaad/sli-sent-large` dataset, then
`kaggle_run eval-si --entry training.sentences.evaluate --datasets baraasaad/sli-sent-large --kernels baraasaad/sli-sent-prep --args "/kaggle/input/sli-sent-large/sentences-large.onnx --protocol SI --part test"`
and the same for `sentences-small.onnx`. This is the **only** SI test run. Expected: two JSON files with `n ≈ 4000`.

- [ ] **Step 2: US protocol.** Train the chosen recipes on US train. Large: `--stage isharah --protocol US` with the same `--init`; small: distill with `--protocol US` and a US teacher. Export both, then evaluate `--protocol US --part test`. Record them. The US and SI models are different checkpoints; only SI models ship.

- [ ] **Step 3: ArabSign (external check).** `arabsign.py` runs on Kaggle CPU:
  1. Take the videos under `ArabSign/Color/*/*/test/` from `osamaalmolike/ssl-data-set` (dataset path via `find_input`).
  2. Extract landmarks with MediaPipe Holistic (`pip install mediapipe==0.10.14`, `mp.solutions.holistic.Holistic(model_complexity=1)`) into the 543-point layout, so `feats.from_holistic` applies with `w = h = 1`, since Holistic gives normalised coords.
  3. Resample to 15 fps with the video's real fps (`cv2.CAP_PROP_FPS`).
  4. Read ArabSign's sentence list (find its annotation file in the dataset; print the dataset tree first) and keep only sentences whose every word is in `vocab["glosses"]` after normalising alef/taa marbuta/yaa (`أإآ→ا`, `ة→ه`, `ى→ي`, matching Isharah's spelling).
  5. If at least 5 sentences remain, report WER and sentence accuracy for both models on them. Otherwise write `{"comparable": false, "overlap": n}`.

  Record whichever outcome occurs; the model card reports it honestly.

- [ ] **Step 4: Write `src/data/metrics.json` → `sentences`**

```json
"sentences": {
  "dataset": "Isharah-1000",
  "large": {"si_test_wer": 0.0, "us_test_wer": 0.0, "arabsign": {}},
  "small": {"si_test_wer": 0.0, "us_test_wer": 0.0, "arabsign": {}},
  "si_test_signers": 4, "latency_large_s": 0.0, "size_small_mb": 0.0
}
```
Fill every value from the Step 1–3 outputs and Task 11 (the zeros above are the schema, not results).

- [ ] **Step 5: Commit** `evaluate.py arabsign.py metrics.json rounds.md`: `Sentences: final evaluation (SI test once, US, ArabSign)`. Report the headline numbers to the user.

---

### Task 13: Server API for the large model

**Files:**
- Create: `server/sentence-api/app.py`, `server/sentence-api/requirements.txt`, `server/sentence-api/Dockerfile`, `server/sentence-api/run.sh`, `server/sentence-api/test_app.py`

**Interfaces:**
- Produces: `POST /api/sentence`, body = float32 LE, length = T × 356 × 4, 8 ≤ T ≤ 450.
  - 200 `{"ids": [int], "glosses": [str], "text": str, "conf": [float], "ms": int, "model": "large-v1"}`
  - 400 bad length; 413 too many frames; 503 busy (queue wait > 4 s); 405 other methods; `GET /api/sentence/health` → `{"ok": true}`
- Env: `SLI_MODEL` (ONNX path), `SLI_VOCAB` (vocab.json), `SLI_PORT` (8000), `SLI_WORKERS` (1)

- [ ] **Step 1: Write the failing tests**

```python
# server/sentence-api/test_app.py
import json, threading, urllib.request, urllib.error
import numpy as np, onnx, pytest
from onnx import TensorProto, helper


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    d = tmp_path_factory.mktemp("m")
    C = 4
    w = helper.make_tensor("W", TensorProto.FLOAT, [356, C], np.random.default_rng(0).normal(size=(356, C)).astype(np.float32).ravel())
    g = helper.make_graph(
        [helper.make_node("Slice", ["feats", "st", "en", "ax", "sp"], ["half"]),
         helper.make_node("MatMul", ["half", "W"], ["z"]), helper.make_node("LogSoftmax", ["z"], ["logprobs"], axis=-1)],
        "t", [helper.make_tensor_value_info("feats", TensorProto.FLOAT, [1, "T", 356])],
        [helper.make_tensor_value_info("logprobs", TensorProto.FLOAT, [1, "T2", C])],
        [w, helper.make_tensor("st", TensorProto.INT64, [1], [0]), helper.make_tensor("en", TensorProto.INT64, [1], [10**9]),
         helper.make_tensor("ax", TensorProto.INT64, [1], [1]), helper.make_tensor("sp", TensorProto.INT64, [1], [2])])
    onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)]), d / "m.onnx")
    json.dump({"glosses": ["a", "b", "c"], "lookup": {"a b": "AB"}, "blank": 0}, open(d / "v.json", "w"))
    import app
    srv = app.make_server(str(d / "m.onnx"), str(d / "v.json"), port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def post(url, arr):
    req = urllib.request.Request(url + "/api/sentence", data=np.asarray(arr, "<f4").tobytes(), method="POST",
                                 headers={"Content-Type": "application/octet-stream"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def test_ok(server):
    code, body = post(server, np.random.default_rng(1).normal(size=(40, 356)))
    assert code == 200 and set(body) >= {"ids", "glosses", "text", "conf", "ms", "model"}


def test_all_zero_frames_ok(server):  # Review Focus 1: nothing detected
    code, body = post(server, np.zeros((40, 356)))
    assert code == 200 and isinstance(body["text"], str)


def test_limits(server):  # Review Focus 3
    assert post(server, np.zeros((451, 356)))[0] == 413
    assert post(server, np.zeros((4, 356)))[0] == 400
    assert post(server, np.zeros(1001))[0] == 400


def test_health(server):
    with urllib.request.urlopen(server + "/api/sentence/health") as r:
        assert json.loads(r.read()) == {"ok": True}
```

- [ ] **Step 2: Run the tests to check they fail**

Run: `cd ~/sli-ai/server/sentence-api && ~/sli-work/kvenv/bin/python -m pytest -q`
Expected: FAIL, `No module named 'app'`.

- [ ] **Step 3: Implement**

```python
# server/sentence-api/app.py
"""Large sentence model behind a tiny HTTP API. Stateless: nothing is logged or stored but counts."""
import json, os, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np, onnxruntime as ort

DIM, MIN_T, MAX_T = 356, 8, 450


def greedy(lp):
    best, probs = lp.argmax(-1), np.exp(lp.max(-1))
    ids, conf, prev, run = [], [], 0, []
    for k, p in zip(best, probs):
        if k != prev and prev != 0:
            conf.append(float(np.mean(run)))
        if k != 0 and k != prev:
            ids.append(int(k)); run = []
        if k != 0:
            run.append(p)
        prev = k
    if prev != 0:
        conf.append(float(np.mean(run)))
    return ids, conf


def make_server(model, vocab_path, port=8000, workers=1):
    o = ort.SessionOptions(); o.intra_op_num_threads = 1; o.inter_op_num_threads = 1
    sess = ort.InferenceSession(model, o)
    vocab = json.load(open(vocab_path, encoding="utf-8"))
    slots = threading.BoundedSemaphore(workers)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # no request logs (privacy)
            pass

        def reply(self, code, body):
            data = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers(); self.wfile.write(data)

        def do_GET(self):
            if self.path == "/api/sentence/health":
                return self.reply(200, {"ok": True})
            self.reply(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/api/sentence":
                return self.reply(404, {"error": "not found"})
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_T * DIM * 4:
                return self.reply(413, {"error": f"at most {MAX_T} frames"})
            if n % (DIM * 4) or n < MIN_T * DIM * 4:
                self.rfile.read(n)
                return self.reply(400, {"error": f"body must be T x {DIM} float32, T >= {MIN_T}"})
            x = np.frombuffer(self.rfile.read(n), "<f4").reshape(1, -1, DIM)
            if not np.isfinite(x).all():
                return self.reply(400, {"error": "non-finite values"})
            if not slots.acquire(timeout=4):
                return self.reply(503, {"error": "busy"})
            try:
                t = time.perf_counter()
                lp = sess.run(["logprobs"], {"feats": x})[0][0]
            finally:
                slots.release()
            ids, conf = greedy(lp)
            glosses = [vocab["glosses"][i - 1] for i in ids]
            key = " ".join(glosses)
            self.reply(200, {"ids": ids, "glosses": glosses, "text": vocab["lookup"].get(key, key),
                             "conf": [round(c, 3) for c in conf], "ms": int((time.perf_counter() - t) * 1000),
                             "model": "large-v1"})

        def do_PUT(self):
            self.reply(405, {"error": "method"})
        do_DELETE = do_PUT

    return ThreadingHTTPServer(("0.0.0.0", port), H)


if __name__ == "__main__":
    make_server(os.environ["SLI_MODEL"], os.environ["SLI_VOCAB"], int(os.environ.get("SLI_PORT", 8000)),
                int(os.environ.get("SLI_WORKERS", 1))).serve_forever()
```

```
# server/sentence-api/requirements.txt
numpy==2.1.3
onnxruntime==1.20.1
```

```dockerfile
# server/sentence-api/Dockerfile
FROM docker.io/library/python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
USER 65534
ENV SLI_MODEL=/models/sentences-large.onnx SLI_VOCAB=/models/sentences-vocab.json SLI_PORT=8000
EXPOSE 8000
CMD ["python", "app.py"]
```

```sh
#!/bin/sh
# server/sentence-api/run.sh: (re)build and start the API; the proxy reaches it on 127.0.0.1:3021.
set -e
cd "$(dirname "$0")"
mkdir -p ~/sli-models
cp ~/sli-work/models/sentences-large.onnx ~/sli-ai/public/models/sentences-vocab.json ~/sli-models/
docker build -t localhost/sli-sentence-api .
docker rm -f sli-sentence-api 2>/dev/null || true
docker run -d --name sli-sentence-api --restart=always --cpus=1 --memory=1g \
  -p 127.0.0.1:3021:8000 -v ~/sli-models:/models:ro,Z localhost/sli-sentence-api
sleep 3 && curl -fsS http://127.0.0.1:3021/api/sentence/health
```

- [ ] **Step 4: Run the tests to check they pass, then deploy**

Run: `cd ~/sli-ai/server/sentence-api && ~/sli-work/kvenv/bin/python -m pytest -q`. Expected: 4 passed.
Then: `chmod +x run.sh && ./run.sh`. Expected: `{"ok": true}`.
Then check real latency through the container:
`~/sli-work/kvenv/bin/python -c "import numpy as np,urllib.request,json,time;b=np.random.default_rng(0).normal(size=(300,356)).astype('<f4').tobytes();t=time.time();r=urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:3021/api/sentence',data=b,method='POST'));print(json.loads(r.read())['ms'],'ms model;',round(time.time()-t,2),'s total')"`
Expected: model time ≤ 2000 ms.

- [ ] **Step 5: Commit**

```bash
git add server/sentence-api
git commit -m "Sentences: server API for the large model (limits, busy queue, no logs)"
```

---

### Task 14: Proxy route + rate limit

**Files:**
- Modify: `~/pricelens/docker/nginx-upstreams/sli.conf`. This lives in the pricelens checkout, outside this repo. Back it up first: `cp sli.conf /tmp/sli.conf.bak-$(date +%s)`.

**Interfaces:**
- Consumes: the API on `127.0.0.1:3021` (Task 13)
- Produces: `https://130-110-124-121.sslip.io/api/sentence` (and `/api/sentence/health`)

- [ ] **Step 1: Write the check that fails now**

Run: `curl -s -o /dev/null -w '%{http_code}\n' https://130-110-124-121.sslip.io/api/sentence/health`
Expected now: `404` (the static site answers).

- [ ] **Step 2: Edit `sli.conf`**

At the top of the file (http context, outside `server {}`):
```nginx
# /api/sentence: at most 20 sentences a minute per address, short bursts allowed.
limit_req_zone $binary_remote_addr zone=sli_sentence:1m rate=20r/m;
```
Inside the `listen 443` server block, before `location /`:
```nginx
    # Large sentence model (server/sentence-api in sli-ai), published on 127.0.0.1:3021.
    location /api/sentence {
        limit_req zone=sli_sentence burst=10 nodelay;
        limit_req_status 429;
        client_max_body_size 700k;
        proxy_pass http://127.0.0.1:3021;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_read_timeout 10s;
        proxy_request_buffering on;
    }
```

- [ ] **Step 3: Test and reload**

Run: `docker exec pricelens-proxy nginx -t && docker exec pricelens-proxy nginx -s reload`
Expected: `syntax is ok`, `test is successful`.
Then: `curl -s https://130-110-124-121.sslip.io/api/sentence/health` → `{"ok": true}`, and
`for i in $(seq 1 15); do curl -s -o /dev/null -w '%{http_code} ' -X POST --data-binary @/dev/null https://130-110-124-121.sslip.io/api/sentence; done; echo`
Expected: a run of `400`s followed by `429`s.
Check that the other sites still answer: `curl -s -o /dev/null -w '%{http_code}\n' https://pricelens.work.gd/` → `200`.

If `nginx -t` fails, restore the backup and reload, then fix.

- [ ] **Step 4: Record it.** The pricelens checkout is a different repo. Commit only if `sli.conf` is tracked there (`git -C ~/pricelens status docker/nginx-upstreams/sli.conf`); otherwise note the change in `server/sentence-api/README` inside sli-ai:

```markdown
# server/sentence-api
Large sentence model API. Deployed with `./run.sh` (container `sli-sentence-api`, 1 CPU, 1 GB,
127.0.0.1:3021). Public route: `location /api/sentence` in
`~/pricelens/docker/nginx-upstreams/sli.conf` (rate limit 20/min/IP, burst 10, body ≤ 700 KB).
```
Commit: `Sentences: document API deployment and proxy route`.

---

### Task 15: App: phone model, API client, recognizer

**Files:**
- Create: `src/recognition/ctc.ts`, `src/recognition/sentenceModel.ts`, `src/recognition/sentence.worker.ts`, `src/recognition/sentenceClient.ts`, `src/recognition/sentences.ts`
- Test: `tests/unit/ctc.test.ts`, `tests/unit/sentences.test.ts`, `tests/replay/sentences.test.ts`

**Interfaces:**
- Consumes: `sentenceFrame`, `resampleFrames`, `SF_DIM` (Task 2); `public/models/sentences-small.onnx`, `sentences-vocab.json` (Task 11); API (Task 13–14)
- Produces:
  - `ctc.ts`: `greedyCtc(logprobs: Float32Array, T: number, C: number): { ids: number[]; conf: number[] }`
  - `sentenceModel.ts`: `interface SentenceVocab { glosses: string[]; lookup: Record<string, string> }`; `glossText(v: SentenceVocab, ids: number[]): { glosses: string[]; text: string }`; `class SentenceModel { static load(base: string): Promise<SentenceModel>; run(feats: Float32Array, T: number): Promise<{ ids: number[]; conf: number[] }> }`
  - `sentenceClient.ts`: `recognizeOnServer(feats: Float32Array, opts?: { url?: string; timeoutMs?: number; fetchImpl?: typeof fetch }): Promise<ServerResult | null>` (null on any failure or timeout)
  - `sentences.ts`: `interface SentenceResult { glosses: string[]; text: string; source: 'server' | 'phone'; final: boolean }`; `interface SentenceOptions { endGapMs: number; minSignMs: number; maxMs: number; liveEveryMs: number; serverTimeoutMs: number }`; `DEFAULT_SENTENCE: SentenceOptions = { endGapMs: 1200, minSignMs: 600, maxMs: 30000, liveEveryMs: 1000, serverTimeoutMs: 4000 }`; `class SentenceRecognizer { constructor(runPhone: (f: Float32Array, T: number) => Promise<{ ids: number[]; conf: number[] }>, vocab: SentenceVocab, events: { onLive?(r: SentenceResult): void; onSentence?(r: SentenceResult): void }, opts?: SentenceOptions, server?: typeof recognizeOnServer); push(tMs: number, raw: RawFrame): Promise<void>; reset(): void }`

- [ ] **Step 1: Write the failing tests**

```ts
// tests/unit/ctc.test.ts
import { describe, expect, it } from 'vitest';
import { greedyCtc } from '../../src/recognition/ctc';

it('collapses repeats and blanks', () => {
  const rows = [0, 2, 2, 0, 2, 3, 3, 0];
  const C = 4;
  const lp = new Float32Array(rows.length * C).fill(Math.log(0.01));
  rows.forEach((k, t) => (lp[t * C + k] = Math.log(0.97)));
  const r = greedyCtc(lp, rows.length, C);
  expect(r.ids).toEqual([2, 2, 3]);
  expect(r.conf).toHaveLength(3);
});
```

```ts
// tests/unit/sentences.test.ts
import { describe, expect, it, vi } from 'vitest';
import type { RawFrame } from '../../src/recognition/features';
import { DEFAULT_SENTENCE, SentenceRecognizer, type SentenceResult } from '../../src/recognition/sentences';

const vocab = { glosses: ['a', 'b'], lookup: { 'a b': 'AB' } };
const hand = (): RawFrame => ({
  pose: Array.from({ length: 33 }, (_, i) => [0.5 + i * 0.001, 0.5, 0, 1] as [number, number, number, number]),
  face: null,
  hands: [{ label: 'Right', lms: Array.from({ length: 21 }, (_, i) => [0.5 + i * 0.01, 0.6, 0] as [number, number, number]) }],
});
const none = (): RawFrame => ({ pose: null, face: null, hands: [] });

async function sign(r: SentenceRecognizer, ms: number, start = 0) {
  let t = start;
  for (; t < start + ms; t += 66) await r.push(t, hand());
  for (let u = t; u < t + DEFAULT_SENTENCE.endGapMs + 200; u += 66) await r.push(u, none());
}

describe('SentenceRecognizer', () => {
  it('uses the server result when it answers', async () => {
    const out: SentenceResult[] = [];
    const server = vi.fn(async () => ({ ids: [1, 2], glosses: ['a', 'b'], text: 'AB', conf: [0.9, 0.9], ms: 5, model: 'large-v1' }));
    const phone = vi.fn(async () => ({ ids: [1], conf: [0.5] }));
    const r = new SentenceRecognizer(phone, vocab, { onSentence: (s) => out.push(s) }, DEFAULT_SENTENCE, server);
    await sign(r, 2000);
    expect(out).toEqual([{ glosses: ['a', 'b'], text: 'AB', source: 'server', final: true }]);
  });

  it('falls back to the phone model when the server fails or times out', async () => { // Review Focus 4
    const out: SentenceResult[] = [];
    const r = new SentenceRecognizer(async () => ({ ids: [1, 2], conf: [0.8, 0.8] }), vocab, { onSentence: (s) => out.push(s) },
      DEFAULT_SENTENCE, async () => null);
    await sign(r, 2000);
    expect(out[0]).toMatchObject({ text: 'AB', source: 'phone', final: true });
  });

  it('ignores movements shorter than minSignMs', async () => { // Review Focus 2
    const server = vi.fn(async () => null);
    const out: SentenceResult[] = [];
    const r = new SentenceRecognizer(async () => ({ ids: [1], conf: [1] }), vocab, { onSentence: (s) => out.push(s) }, DEFAULT_SENTENCE, server);
    await sign(r, 300);
    expect(out).toHaveLength(0);
    expect(server).not.toHaveBeenCalled();
  });

  it('finalises at maxMs even if the hands never drop', async () => { // Review Focus 3
    const out: SentenceResult[] = [];
    const lens: number[] = [];
    const r = new SentenceRecognizer(async () => ({ ids: [1], conf: [1] }), vocab, { onSentence: (s) => out.push(s) }, DEFAULT_SENTENCE,
      async (f) => { lens.push(f.length / 356); return null; });
    for (let t = 0; t < 31000; t += 66) await r.push(t, hand());
    expect(out.length).toBe(1);
    expect(lens[0]).toBeLessThanOrEqual(450);
  });
});
```

`recognizeOnServer` gets its own small test in the same file, with a hanging `fetchImpl` that must resolve `null` after `timeoutMs`:

```ts
import { recognizeOnServer } from '../../src/recognition/sentenceClient';
it('server client times out to null', async () => {
  const hang = (() => new Promise(() => {})) as unknown as typeof fetch;
  const t = Date.now();
  expect(await recognizeOnServer(new Float32Array(356 * 10), { fetchImpl: hang, timeoutMs: 100 })).toBeNull();
  expect(Date.now() - t).toBeLessThan(1000);
});
```

- [ ] **Step 2: Run them to check they fail**

Run: `npx vitest run tests/unit/ctc.test.ts tests/unit/sentences.test.ts`
Expected: FAIL, modules not found.

- [ ] **Step 3: Implement**

```ts
// src/recognition/ctc.ts
/** Greedy CTC decoding (blank = 0), as training/sentences/ctc.py. */
export function greedyCtc(lp: Float32Array, T: number, C: number): { ids: number[]; conf: number[] } {
  const ids: number[] = [], conf: number[] = [];
  let prev = 0, run: number[] = [];
  for (let t = 0; t < T; t++) {
    let k = 0, best = -Infinity;
    for (let c = 0; c < C; c++) if (lp[t * C + c] > best) { best = lp[t * C + c]; k = c; }
    if (k !== prev && prev !== 0) conf.push(run.reduce((a, b) => a + b, 0) / run.length);
    if (k !== 0 && k !== prev) { ids.push(k); run = []; }
    if (k !== 0) run.push(Math.exp(best));
    prev = k;
  }
  if (prev !== 0) conf.push(run.reduce((a, b) => a + b, 0) / run.length);
  return { ids, conf };
}
```

```ts
// src/recognition/sentenceModel.ts
import * as ort from 'onnxruntime-web/wasm';
import { greedyCtc } from './ctc';
import { SF_DIM } from './sentenceFeatures';

export interface SentenceVocab { glosses: string[]; lookup: Record<string, string> }

export function glossText(v: SentenceVocab, ids: number[]): { glosses: string[]; text: string } {
  const glosses = ids.map((i) => v.glosses[i - 1]).filter(Boolean);
  const key = glosses.join(' ');
  return { glosses, text: v.lookup[key] ?? key };
}

export class SentenceModel {
  private constructor(private session: ort.InferenceSession) {}

  static async load(base: string): Promise<SentenceModel> {
    ort.env.wasm.numThreads = 1;
    const session = await ort.InferenceSession.create(`${base}models/sentences-small.onnx`, {
      executionProviders: ['wasm'], graphOptimizationLevel: 'all',
    });
    return new SentenceModel(session);
  }

  async run(feats: Float32Array, T: number) {
    const out = await this.session.run({ feats: new ort.Tensor('float32', feats, [1, T, SF_DIM]) });
    const lp = out.logprobs;
    const [, T2, C] = lp.dims;
    return greedyCtc(lp.data as Float32Array, T2, C);
  }
}
```

```ts
// src/recognition/sentence.worker.ts
/// <reference lib="webworker" />
// The phone sentence model runs in its own worker, like the word model.
import { SentenceModel } from './sentenceModel';

export type SentenceWorkerRequest = { type: 'init'; base: string } | { type: 'run'; id: number; feats: Float32Array; T: number };
export type SentenceWorkerReply =
  | { type: 'ready' }
  | { type: 'error'; message: string; id?: number }
  | { type: 'result'; id: number; ids: number[]; conf: number[]; ms: number };

let model: SentenceModel | null = null;
const post = (m: SentenceWorkerReply) => self.postMessage(m);

self.onmessage = async (e: MessageEvent<SentenceWorkerRequest>) => {
  const m = e.data;
  try {
    if (m.type === 'init') { model = await SentenceModel.load(m.base); post({ type: 'ready' }); }
    else {
      const start = performance.now();
      const r = await model!.run(m.feats, m.T);
      post({ type: 'result', id: m.id, ...r, ms: performance.now() - start });
    }
  } catch (err) {
    post({ type: 'error', message: err instanceof Error ? err.message : String(err), id: m.type === 'run' ? m.id : undefined });
  }
};
```

```ts
// src/recognition/sentenceClient.ts
// The large model on the SLI server. Only SF1 numbers are sent, never camera images.
export interface ServerResult { ids: number[]; glosses: string[]; text: string; conf: number[]; ms: number; model: string }

export async function recognizeOnServer(
  feats: Float32Array,
  opts: { url?: string; timeoutMs?: number; fetchImpl?: typeof fetch } = {},
): Promise<ServerResult | null> {
  const { url = '/api/sentence', timeoutMs = 4000, fetchImpl = fetch } = opts;
  if (typeof navigator !== 'undefined' && navigator.onLine === false) return null;
  const ctrl = new AbortController();
  const timer = new Promise<null>((resolve) => setTimeout(() => { ctrl.abort(); resolve(null); }, timeoutMs));
  const req = (async () => {
    try {
      const res = await fetchImpl(url, { method: 'POST', body: feats.buffer.slice(0) as ArrayBuffer, signal: ctrl.signal,
        headers: { 'Content-Type': 'application/octet-stream' } });
      return res.ok ? ((await res.json()) as ServerResult) : null;
    } catch {
      return null;
    }
  })();
  return Promise.race([req, timer]);
}
```

```ts
// src/recognition/sentences.ts
// Sentences mode: buffer a signing stretch (hands visible), show phone-model guesses while
// signing, and when the hands drop (or 30 s pass) ask the large server model, falling back to
// the phone model.
import type { RawFrame } from './features';
import { SF_DIM, resampleFrames, sentenceFrame } from './sentenceFeatures';
import { glossText, type SentenceVocab } from './sentenceModel';
import { recognizeOnServer } from './sentenceClient';

export interface SentenceResult { glosses: string[]; text: string; source: 'server' | 'phone'; final: boolean }
export interface SentenceOptions { endGapMs: number; minSignMs: number; maxMs: number; liveEveryMs: number; serverTimeoutMs: number }
export const DEFAULT_SENTENCE: SentenceOptions = { endGapMs: 1200, minSignMs: 600, maxMs: 30000, liveEveryMs: 1000, serverTimeoutMs: 4000 };
const MAX_FRAMES = 450;
const MIN_FRAMES = 8;

type Run = (f: Float32Array, T: number) => Promise<{ ids: number[]; conf: number[] }>;

export class SentenceRecognizer {
  private frames: Float32Array[] = [];
  private times: number[] = [];
  private firstHand = -1;
  private lastHand = -1;
  private lastLive = -1;
  private busy = false;

  constructor(
    private runPhone: Run,
    private vocab: SentenceVocab,
    private events: { onLive?(r: SentenceResult): void; onSentence?(r: SentenceResult): void },
    private opts: SentenceOptions = DEFAULT_SENTENCE,
    private server: typeof recognizeOnServer = recognizeOnServer,
  ) {}

  reset() {
    this.frames = []; this.times = []; this.firstHand = this.lastHand = this.lastLive = -1;
  }

  private window(): { feats: Float32Array; T: number } {
    let feats = resampleFrames(this.frames, this.times);
    let T = feats.length / SF_DIM;
    if (T > MAX_FRAMES) { feats = feats.slice((T - MAX_FRAMES) * SF_DIM); T = MAX_FRAMES; }
    return { feats, T };
  }

  async push(t: number, raw: RawFrame) {
    const seen = raw.hands.length > 0;
    if (seen) {
      if (this.firstHand < 0) this.firstHand = t;
      this.lastHand = t;
    }
    if (this.firstHand < 0) return; // nothing to buffer before the first hand
    this.frames.push(sentenceFrame(raw));
    this.times.push(t);
    const signedFor = this.lastHand - this.firstHand;
    const ended = !seen && t - this.lastHand >= this.opts.endGapMs;
    const full = t - this.firstHand >= this.opts.maxMs;
    if (ended || full) {
      const long = signedFor >= this.opts.minSignMs;
      const { feats, T } = this.window();
      this.reset();
      if (long && T >= MIN_FRAMES) await this.finish(feats, T);
      return;
    }
    if (seen && !this.busy && t - this.lastLive >= this.opts.liveEveryMs && signedFor >= this.opts.minSignMs) {
      this.lastLive = t;
      this.busy = true;
      const { feats, T } = this.window();
      try {
        const r = await this.runPhone(feats, T);
        this.events.onLive?.({ ...glossText(this.vocab, r.ids), source: 'phone', final: false });
      } finally {
        this.busy = false;
      }
    }
  }

  private async finish(feats: Float32Array, T: number) {
    const s = await this.server(feats, { timeoutMs: this.opts.serverTimeoutMs });
    if (s) {
      this.events.onSentence?.({ glosses: s.glosses, text: s.text, source: 'server', final: true });
      return;
    }
    const r = await this.runPhone(feats, T);
    this.events.onSentence?.({ ...glossText(this.vocab, r.ids), source: 'phone', final: true });
  }
}
```

In the tests `onLive` fires too, but they only collect `onSentence`, so each test sees exactly one final result.

- [ ] **Step 4: Replay test with the real phone model**

```ts
// tests/replay/sentences.test.ts
// Browser path (SF1 in TS + phone ONNX + greedy CTC) equals the Python export on the parity cases.
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import * as ort from 'onnxruntime-node';
import type { RawFrame } from '../../src/recognition/features';
import { SF_DIM, resampleFrames, sentenceFrame } from '../../src/recognition/sentenceFeatures';
import { greedyCtc } from '../../src/recognition/ctc';

const cases: { name: string; raws: RawFrame[]; t: number[] }[] = JSON.parse(readFileSync(new URL('../fixtures/sentence-parity.json', import.meta.url), 'utf8'));
const expected: { name: string; logprobs_head: number[]; ids: number[] }[] = JSON.parse(readFileSync(new URL('../fixtures/sentence-model-parity.json', import.meta.url), 'utf8'));

describe('phone sentence model matches Python', () => {
  it.each(cases.map((c, i) => [c.name, c, expected[i]] as const))('%s', async (_n, c, e) => {
    const feats = resampleFrames(c.raws.map(sentenceFrame), c.t);
    const T = feats.length / SF_DIM;
    const s = await ort.InferenceSession.create(new URL('../../public/models/sentences-small.onnx', import.meta.url).pathname);
    const lp = (await s.run({ feats: new ort.Tensor('float32', feats, [1, T, SF_DIM]) })).logprobs;
    e.logprobs_head.forEach((v, i) => expect((lp.data as Float32Array)[i]).toBeCloseTo(v, 2));
    expect(greedyCtc(lp.data as Float32Array, lp.dims[1], lp.dims[2]).ids).toEqual(e.ids);
  });
});
```

This file needs no training cache, so it runs in the normal `npm test` (the existing `replay.test.ts` is gated by `SLI_REPLAY`; this one is not).

- [ ] **Step 5: Run everything**

Run: `npx vitest run && npx tsc --noEmit`
Expected: all unit and replay tests pass (the old ones too); no type errors.

- [ ] **Step 6: Commit**

```bash
git add src/recognition/ctc.ts src/recognition/sentenceModel.ts src/recognition/sentence.worker.ts src/recognition/sentenceClient.ts src/recognition/sentences.ts tests/unit/ctc.test.ts tests/unit/sentences.test.ts tests/replay/sentences.test.ts
git commit -m "Sentences: phone model worker, server client with fallback, recognizer"
```

---

### Task 16: App: "Sentences" mode in the UI + e2e

**Files:**
- Modify: `src/recognition/interpreter.ts:17` (`SignMode`), `src/recognition/engine.ts` (`setMode`, frame loop around line 246–273, new events), `src/ui/capture.ts` (mode buttons, result display), `src/i18n.ts` (strings, both languages), `src/styles.css` (result line)
- Create: `tests/e2e/sentences.spec.ts`, `tests/fixtures/camera-sentence.y4m`

**Interfaces:**
- Consumes: `SentenceRecognizer`, `SentenceResult`, `sentence.worker.ts` messages (Task 15)
- Produces: `SignMode = 'auto' | 'words' | 'letters' | 'sentences'`; `EngineEvents.onSentenceLive?(r: SentenceResult)`, `EngineEvents.onSentence?(r: SentenceResult)`; `window.__sliSentences: SentenceResult[]` when `?debug`

- [ ] **Step 1: Write the failing e2e test**

Make the camera fixture from one ArabSign test video whose sentence is in the vocabulary (Task 12 Step 3 lists the overlapping sentences; if there were none, take any ArabSign test video). Use the same 4× slowdown as the other e2e cameras:
```bash
ffmpeg -y -i <arabsign-test-video.mp4> -vf "setpts=4*PTS,fps=30,scale=640:480:force_original_aspect_ratio=decrease,pad=640:480:(ow-iw)/2:(oh-ih)/2" -pix_fmt yuv420p -t 60 tests/fixtures/camera-sentence.y4m
```

```ts
// tests/e2e/sentences.spec.ts
import { expect, test } from '@playwright/test';

test.use({ launchOptions: { args: ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream',
  `--use-file-for-fake-video-capture=${new URL('../fixtures/camera-sentence.y4m', import.meta.url).pathname}`] } });

test('sentences mode produces a sentence from the server model', async ({ page }) => {
  const api: number[] = [];
  page.on('response', (r) => { if (r.url().includes('/api/sentence')) api.push(r.status()); });
  await page.goto('/?debug&timescale=4#/sign');
  await page.getByRole('button', { name: /جمل|Sentences/ }).click();
  await page.getByRole('button', { name: /تشغيل الكاميرا|Start camera/ }).click();
  await expect.poll(async () => page.evaluate(() => (window as any).__sliSentences?.length ?? 0), { timeout: 240_000 }).toBeGreaterThan(0);
  const s = await page.evaluate(() => (window as any).__sliSentences[0]);
  expect(s.final).toBe(true);
  expect(s.text.length).toBeGreaterThan(0);
  expect(['server', 'phone']).toContain(s.source);
  const frames = await page.evaluate(() => (window as any).__sliFrames as { raw: { pose: unknown; face: unknown } }[]);
  expect(frames.some((f) => f.raw.pose) && frames.some((f) => f.raw.face)).toBe(true); // body landmarks requested in this mode
});
```

The e2e server is `vite preview`, which has no `/api/sentence`. Add a preview proxy to `vite.config.ts` (`preview.proxy['/api'] = 'http://127.0.0.1:3021'`, same for `server.proxy`) so the test exercises the real API on the server. The camera button labels must match the real i18n strings, so check `t().startCamera` in `src/i18n.ts` and adjust the regexes.

Run inside the usual Playwright podman image (see the existing e2e setup: `mcr.microsoft.com/playwright:v1.63.0-noble`):
`npx playwright test tests/e2e/sentences.spec.ts`
Expected: FAIL (no Sentences button).

- [ ] **Step 2: Implement the mode**

1. `interpreter.ts`: `export type SignMode = 'auto' | 'words' | 'letters' | 'sentences';`. The Interpreter never receives frames in this mode (next item). Its `setMode` just stores it.
2. `engine.ts`:
   - Add to `EngineEvents`: `onSentenceLive?(r: SentenceResult): void; onSentence?(r: SentenceResult): void;`
   - Add a lazily created recognizer, built the first time the mode is `'sentences'`:
     ```ts
     private sentences: Promise<SentenceRecognizer> | null = null;
     private startSentences(base: string): Promise<SentenceRecognizer> {
       const worker = new Worker(new URL('./sentence.worker.ts', import.meta.url), { type: 'module' });
       const pending = new Map<number, (r: { ids: number[]; conf: number[] }) => void>();
       let next = 0;
       const run = (feats: Float32Array, T: number) => new Promise<{ ids: number[]; conf: number[] }>((resolve) => {
         const id = next++; pending.set(id, resolve);
         const req: SentenceWorkerRequest = { type: 'run', id, feats, T };
         worker.postMessage(req, [feats.buffer]);
       });
       return Promise.all([
         fetch(`${base}models/sentences-vocab.json`).then((r) => r.json() as Promise<SentenceVocab>),
         new Promise<void>((resolve, reject) => {
           worker.onmessage = (e: MessageEvent<SentenceWorkerReply>) => {
             const m = e.data;
             if (m.type === 'ready') resolve();
             else if (m.type === 'result') { pending.get(m.id)?.(m); pending.delete(m.id); }
             else if (m.id !== undefined) { pending.get(m.id)?.({ ids: [], conf: [] }); pending.delete(m.id); }
             else reject(new Error(m.message));
           };
           const init: SentenceWorkerRequest = { type: 'init', base };
           worker.postMessage(init);
         }),
       ]).then(([vocab]) => new SentenceRecognizer(run, vocab, {
         onLive: (r) => this.events.onSentenceLive?.(r),
         onSentence: (r) => { sentenceLog?.push(r); this.events.onSentence?.(r); },
       }, sentenceOptions()));
     }
     ```
     with, at module level next to `debugFrames`:
     `const sentenceLog: SentenceResult[] | null = params.has('debug') ? [] : null; if (sentenceLog) Object.assign(window, { __sliSentences: sentenceLog });`
     and `const sentenceOptions = (): SentenceOptions => DEFAULT_SENTENCE;`.
     Timing under `?timescale=N` (slowed test cameras) is handled by giving the recognizer real-speed time: frames are pushed as `push(t / timescale, raw)`. Its options therefore stay unscaled, and the model sees signing at its true speed.
   - `setMode(mode)`: after the existing lines, `if (mode === 'sentences') this.sentences ??= this.startSentences(this.base); else void this.sentences?.then((r) => r.reset());`. Use the `base` the engine already passes to `startLandmarks`. Keep it on `this.base` if it is not already stored.
   - In the frame loop, where `needsBody` is computed (≈ line 246): `const needsBody = this.mode === 'sentences' || (this.interpreter?.needsBody ?? this.mode !== 'letters');`
   - Where frames go to the interpreter (≈ line 273): `if (this.mode === 'sentences') void this.sentences?.then((r) => r.push(t / timescale, raw)); else void this.interpreter?.push(t, raw);`
3. `capture.ts`: add `'sentences'` to the array literal of modes that builds `this.modeButtons`. Add
   `private sentenceOut = h('p', { class: 'sentence-result', 'aria-live': 'polite' });` to the capture element after `this.hint`. In the `Engine` events:
   `onSentenceLive: (r) => { this.live.textContent = r.text; }`,
   `onSentence: (r) => { this.live.textContent = ''; this.sentenceOut.textContent = r.text; this.sentenceOut.dataset.source = r.source; this.onSentence?.(r.text); }`.
   Add an optional constructor callback `onSentence?: (text: string) => void` after the existing `onCommit` callback, and hide `sentenceOut` when the mode is not `'sentences'`.
4. `sign.ts`: pass `(text) => { spoken = text; speakBtn.disabled = !text; if (autoSpeak) void speak(text); }` as `onSentence`. Make the speak button speak `spoken` when the mode is sentences; keep `sentence.text()` otherwise.
5. `i18n.ts`, both `ar` and `en`:
   - `modes.sentences`: `'جمل'` / `'Sentences'`
   - `modeHints.sentences`: `'أشِر بجملة كاملة بشكل طبيعي، ثم أنزل يديك لتظهر الترجمة. تعمل الجمل المعروفة في لغة الإشارة السعودية.'` / `'Sign a whole sentence naturally, then lower your hands to see the translation. Works for known Saudi Sign Language sentences.'`
   - `sentenceFrom: { server: 'ترجمة الخادم', phone: 'ترجمة على الجهاز' }` / `{ server: 'Server translation', phone: 'On-device translation' }`. Show it as a small label after the text using `dataset.source`.
6. `styles.css`: `.sentence-result { font-size: 1.4rem; min-height: 2em; margin: .5rem 0; }` `.sentence-result:empty { display: none; }`.

- [ ] **Step 3: Run the tests**

Run: `npx vitest run && npx tsc --noEmit && npx playwright test tests/e2e/sentences.spec.ts tests/e2e/app.spec.ts`
Expected: all pass. `app.spec.ts` checks that the words mode is not broken.

- [ ] **Step 4: Build and deploy the site**

`npm run build && cp -r dist/. ~/sli-site/html/ && chmod -R a+rX ~/sli-site/html`, then open https://130-110-124-121.sslip.io/#/sign, pick "Sentences" and check the network tab shows `/api/sentence` 200. Tell the user it is live.

- [ ] **Step 5: Commit**

```bash
git add src tests/e2e/sentences.spec.ts tests/fixtures/camera-sentence.y4m vite.config.ts
git commit -m "Sentences mode: live phone guesses, server sentence, offline fallback"
```

---

### Task 17: Hugging Face release + docs

**Files:**
- Create: `hf/README.md` (model card), `hf/upload.py`
- Modify: `README.md` (project), `training/README.md` (Sentences section)

**Interfaces:**
- Consumes: `~/sli-work/models/sentences-large.onnx`, `public/models/sentences-small.onnx`, `public/models/sentences-vocab.json`, `src/data/metrics.json` (`sentences`), `training/sentences/feats.py` (feature spec)
- Produces: public HF model repo `<hf-user>/sli-saudi-sign-sentences`, with the name confirmed by the user

- [ ] **Step 1: Ask the user** for a Hugging Face **write** token and the repo name/visibility. Store the token like the Kaggle one (`~/.cache/huggingface/token`, mode 600, via stdin, never echoed). Wait for the answer before Step 3.

- [ ] **Step 2: Write the model card `hf/README.md`**

It needs:
- YAML header: `license: cc-by-nc-sa-4.0`, `language: [ar]`, `tags: [sign-language, saudi-sign-language, cslr, ctc, onnx, mediapipe]`, `library_name: onnx`
- **What it does:** landmarks → gloss sequence → text lookup; two files (large for servers, small ≤ 20 MB for phones).
- **Input spec:** SF1 exactly as in `feats.py` (layout, anchors, scales, 15 fps, presence flags), plus a 30-line reference Python snippet copied from `feats.from_app_raw` + `resample`.
- **Results table:** from `metrics.json`, WER on SI test (4 unseen signers), US test and ArabSign, for both models, next to the Isharah paper's baselines from `rounds.md`.
- **Limitations:** 685 glosses only; Saudi Sign Language; sentence-level lookup text only for known sentences; not for commercial use; not a substitute for a human interpreter in medical, legal or emergency settings.
- **Training:** data, stages, augmentation, compute (Kaggle GPU hours from `rounds.md`).
- **Credits and citations:** Isharah (BibTeX from its README), KArSL, yousefelkilany/Word-level-Arabic-Sign-language (MIT; the earlier word model and preprocessing), pose-format, MediaPipe.

- [ ] **Step 3: Upload**

```python
# hf/upload.py
import os, sys
from huggingface_hub import HfApi
repo = sys.argv[1]
api = HfApi()
api.create_repo(repo, repo_type="model", exist_ok=True, private="--private" in sys.argv)
for src, dst in [("hf/README.md", "README.md"), (os.path.expanduser("~/sli-work/models/sentences-large.onnx"), "sentences-large.onnx"),
                 ("public/models/sentences-small.onnx", "sentences-small.onnx"), ("public/models/sentences-vocab.json", "vocab.json"),
                 ("training/sentences/feats.py", "sf1_features.py"), ("training/face_idx.json", "face_idx.json"),
                 ("src/data/metrics.json", "metrics.json")]:
    api.upload_file(path_or_fileobj=src, path_in_repo=dst, repo_id=repo)
print("https://huggingface.co/" + repo)
```
Run: `~/.local/bin/uv pip install -q -p ~/sli-work/kvenv huggingface_hub && ~/sli-work/kvenv/bin/python hf/upload.py <hf-user>/sli-saudi-sign-sentences`
Expected: the URL prints; the page shows the card and 7 files.
Check: `sf1_features.py` imports `conventions.json`, so either upload it too or inline the two values. Download the repo into a temp dir and run the snippet from the card on the parity fixture; the output must match `sentence-parity.json`.

- [ ] **Step 4: Docs.** In `training/README.md`, add a "Sentences" section with the file table (as for the word model), the results table and the Kaggle commands. In `README.md`, add the Sentences mode and the HF link.

- [ ] **Step 5: Commit** `hf/ README.md training/README.md`: `Sentences: Hugging Face model card and release`. Ask the user whether to push `main` to GitHub.

---

## Self-review notes

- **Spec coverage:**

  | Spec section | Task(s) |
  | --- | --- |
  | Goal: two models + HF | 8–11, 17 |
  | Data | 5 (Isharah + KArSL), 12 (ArabSign) |
  | Features (SF1) | 1–3 |
  | Models (large/small/latency) | 6, 9–11 |
  | Infra (Kaggle) | 4 |
  | Stopping rule | 9 |
  | Evaluation (SI once, US, ArabSign, replay) | 12, 15 |
  | Gloss→text | 5 (`lookup`), 13, 15 |
  | Server API | 13–14 |
  | App | 15–16 |
  | HF | 17 |
  | Out of scope | respected: no video models, no translation model |

- **Types:**
  - `SentenceVocab` / `glossText` are defined in Task 15 and used in 16.
  - `SentenceResult` is defined in 15 and used in 16.
  - `SF_DIM` / `resampleFrames` come from Task 2.
  - The ONNX I/O names `feats` / `logprobs` are the same in Tasks 11, 13 and 15.
  - Blank = 0 everywhere.
- **Review Focus:** Each item has a test in its owning task:
  - hand or face missing: Tasks 2, 13
  - movements that are too short: Task 15
  - signing longer than 30 s: Tasks 13, 15
  - server down: Task 15
  - irregular frame rate: Tasks 1, 2
