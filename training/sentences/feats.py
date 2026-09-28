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
SWAP = [1, 0, 3, 2, 5, 4]  # pose subset left <-> right


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
    """Signer side for each hand (True = left): nearest pose wrist (15 = left, 16 = right)."""
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
        out[o:o + 42] = slot.ravel()  # already wrist-relative and hand-scaled
        out[O_PRES + (2 if left else 3)] = 1
    return out


def resample(frames, t_ms, fps=FPS):
    """Linear interpolation of frames taken at t_ms onto a uniform fps grid starting at t_ms[0]."""
    frames = np.asarray(frames, np.float32)
    if len(frames) == 1:
        return frames.copy()
    t = np.asarray(t_ms, np.float64) - t_ms[0]
    step = 1000.0 / fps
    grid = np.arange(int(t[-1] / step) + 1) * step
    idx = np.clip(np.searchsorted(t, grid, side="right") - 1, 0, len(t) - 2)
    t0, t1 = t[idx], t[idx + 1]
    a = np.clip((grid - t0) / np.maximum(t1 - t0, 1e-9), 0, 1)[:, None]
    return ((1 - a) * frames[idx] + a * frames[idx + 1]).astype(np.float32)


def mirror(frames):
    """Left-right mirror of a signer: arms and hands mirrored and swapped; the face is kept."""
    x = np.array(frames, np.float32, copy=True)
    pose = x[:, O_POSE:O_POSE + 12].reshape(-1, 6, 2)
    new = pose[:, SWAP].copy()
    new -= new[:, :1]  # re-anchor on the new left shoulder (the old right shoulder)
    new[..., 0] *= -1
    x[:, O_POSE:O_POSE + 12] = new.reshape(-1, 12) * x[:, [O_PRES]]
    lh, rh = x[:, O_LH:O_LH + 42].copy(), x[:, O_RH:O_RH + 42].copy()
    lh[:, 0::2] *= -1
    rh[:, 0::2] *= -1
    x[:, O_LH:O_LH + 42], x[:, O_RH:O_RH + 42] = rh, lh
    x[:, [O_PRES + 2, O_PRES + 3]] = x[:, [O_PRES + 3, O_PRES + 2]]
    return x
