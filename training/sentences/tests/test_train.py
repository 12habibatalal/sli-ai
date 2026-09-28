import numpy as np
import torch

from training.sentences import feats as F
from training.sentences import model as M
from training.sentences import train as T


def test_tiny_model_overfits_four_sentences():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    protos = {g: rng.normal(0, 1, (6, F.FEAT_DIM)).astype(np.float32) for g in (1, 2, 3)}
    feats, items = {}, []
    for k, seq in enumerate([[1, 2], [2, 3], [3, 1, 2], [1]]):
        feats[str(k)] = np.concatenate([protos[g] for g in seq])
        items.append((str(k), seq))
    m = M.SignCTC(4, d=64, layers=2, heads=4, ff=128, conv=3, dropout=0.0)
    T.fit(m, feats, items, feats, items, epochs=150, lr=3e-3, max_frames=200, device="cpu", augment=False)
    assert T.evaluate_split(m, feats, items, "cpu") == 0.0
