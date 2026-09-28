// Browser port of training/sentences/feats.py: sentence features v1 (SF1) for the sentence
// models. Parity with Python: tests/unit/sentenceFeatures.test.ts.
//
// Layout per frame: pose 6 | face 128 | left hand 21 | right hand 21 as (x, y), then presence
// flags (pose, face, left hand, right hand). Hands are stored by the signer's own side.

import faceIdx from '../../training/face_idx.json';
import conv from '../../training/sentences/conventions.json';
import type { RawFrame } from './features';

export const SF_DIM = 356;
export const SF_FPS = 15;

const FACE_IDX: readonly number[] = faceIdx.face.slice(0, 128); // no iris points
const POSE_IDX = [11, 12, 13, 14, 15, 16];
const O_FACE = 12;
const O_LH = 268;
const O_RH = 310;
const O_PRES = 352;

type XY = ArrayLike<number>;
const dist = (a: XY, b: XY) => Math.max(Math.hypot(a[0] - b[0], a[1] - b[1]), 1e-6);

function write(out: Float32Array, o: number, pts: XY[], ax: number, ay: number, s: number) {
  for (let i = 0; i < pts.length; i++) {
    out[o + 2 * i] = (pts[i][0] - ax) / s;
    out[o + 2 * i + 1] = (pts[i][1] - ay) / s;
  }
}

/** Signer side of each hand (true = left): the nearest pose wrist; MediaPipe's label flips. */
function assign(raw: RawFrame): boolean[] {
  const hands = raw.hands.slice(0, 2);
  if (!raw.pose) return hands.map((h) => (h.label === 'Left') === conv.app_left_label_is_signer_left);
  const lw = raw.pose[15];
  const rw = raw.pose[16];
  const d = (p: XY, q: XY) => Math.hypot(p[0] - q[0], p[1] - q[1]);
  if (hands.length === 2) {
    const a = hands[0].lms[0];
    const b = hands[1].lms[0];
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
  if (tMs.length === 1) return 1;
  return Math.floor((tMs[tMs.length - 1] - tMs[0]) / (1000 / fps)) + 1;
}

/** Linear interpolation onto a uniform fps grid starting at tMs[0], as feats.resample. */
export function resampleFrames(frames: Float32Array[], tMs: number[], fps = SF_FPS): Float32Array {
  const n = resampledLength(tMs, fps);
  const out = new Float32Array(n * SF_DIM);
  if (frames.length === 1) {
    out.set(frames[0]);
    return out;
  }
  const step = 1000 / fps;
  let j = 0;
  for (let g = 0; g < n; g++) {
    const tg = g * step;
    // last j with t[j] <= tg, clipped to [0, T-2] (np.searchsorted side="right" - 1)
    while (j < frames.length - 2 && tMs[j + 1] - tMs[0] <= tg) j++;
    const t0 = tMs[j] - tMs[0];
    const t1 = tMs[j + 1] - tMs[0];
    const a = Math.min(Math.max((tg - t0) / Math.max(t1 - t0, 1e-9), 0), 1);
    const f0 = frames[j];
    const f1 = frames[j + 1];
    const o = g * SF_DIM;
    for (let k = 0; k < SF_DIM; k++) out[o + k] = (1 - a) * f0[k] + a * f1[k];
  }
  return out;
}
