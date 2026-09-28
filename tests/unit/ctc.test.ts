import { expect, it } from 'vitest';
import { greedyCtc } from '../../src/recognition/ctc';

it('collapses repeats and drops blanks, like training/sentences/ctc.py', () => {
  const rows = [0, 2, 2, 0, 2, 3, 3, 0];
  const C = 4;
  const lp = new Float32Array(rows.length * C).fill(Math.log(0.01));
  rows.forEach((k, t) => (lp[t * C + k] = Math.log(0.97)));
  const r = greedyCtc(lp, rows.length, C);
  expect(r.ids).toEqual([2, 2, 3]);
  expect(r.conf).toHaveLength(3);
  r.conf.forEach((c) => expect(c).toBeCloseTo(0.97, 5));
});
