// Parity with the Python SF1 features (training/sentences/export_parity.py).
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import type { RawFrame } from '../../src/recognition/features';
import { SF_DIM, resampleFrames, resampledLength, sentenceFrame } from '../../src/recognition/sentenceFeatures';

interface Case {
  name: string;
  raws: RawFrame[];
  t: number[];
  frames: number[][];
}

const cases: Case[] = JSON.parse(readFileSync(new URL('../fixtures/sentence-parity.json', import.meta.url), 'utf8'));

describe('SF1 features match Python', () => {
  it.each(cases.map((c) => [c.name, c] as const))('%s', (_name, c) => {
    const flat = resampleFrames(c.raws.map(sentenceFrame), c.t);
    expect(resampledLength(c.t)).toBe(c.frames.length);
    expect(flat.length).toBe(c.frames.length * SF_DIM);
    c.frames.forEach((row, i) => row.forEach((v, k) => expect(flat[i * SF_DIM + k]).toBeCloseTo(v, 4)));
  });

  it('an empty frame is all zeros, never NaN', () => {
    const f = sentenceFrame({ pose: null, face: null, hands: [] });
    expect(f.length).toBe(SF_DIM);
    expect(f.every((v) => v === 0)).toBe(true);
  });
});
