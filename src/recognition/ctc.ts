/** Greedy CTC decoding (blank = 0), as training/sentences/ctc.py. `lp` is [T, C] log-probs. */
export function greedyCtc(lp: Float32Array, T: number, C: number): { ids: number[]; conf: number[] } {
  const ids: number[] = [];
  const conf: number[] = [];
  const mean = (a: number[]) => a.reduce((s, v) => s + v, 0) / a.length;
  let prev = 0;
  let run: number[] = [];
  for (let t = 0; t < T; t++) {
    let k = 0;
    let best = -Infinity;
    for (let c = 0; c < C; c++) {
      if (lp[t * C + c] > best) {
        best = lp[t * C + c];
        k = c;
      }
    }
    if (k !== prev && prev !== 0) conf.push(mean(run));
    if (k !== 0 && k !== prev) {
      ids.push(k);
      run = [];
    }
    if (k !== 0) run.push(Math.exp(best));
    prev = k;
  }
  if (prev !== 0) conf.push(mean(run));
  return { ids, conf };
}
