"""Greedy CTC decoding (blank = 0) and word error rate. src/recognition/ctc.ts mirrors greedy()."""
import numpy as np


def greedy(logprobs):
    """[T, C] log-probs -> (ids without blanks and repeats, mean max-prob of each emitted token)."""
    best = logprobs.argmax(-1)
    probs = np.exp(logprobs.max(-1))
    ids, conf, prev, run = [], [], 0, []
    for k, p in zip(best, probs):
        if k != prev and prev != 0:
            conf.append(float(np.mean(run)))
        if k != 0 and k != prev:
            ids.append(int(k))
            run = []
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
    """Total edit distance over total reference length."""
    return sum(_edits(r, h) for r, h in zip(refs, hyps)) / max(sum(len(r) for r in refs), 1)
