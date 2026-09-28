import numpy as np
import torch

from training.sentences import evaluate as V
from training.sentences import export as E
from training.sentences import model as M

ARCH = dict(d=32, layers=1, heads=4, ff=64, conv=3)


def test_evaluate_counts_only_the_requested_split(tmp_path):
    torch.manual_seed(0)
    m = M.SignCTC(3, **ARCH).eval()
    torch.save({"state": m.state_dict(), "arch": ARCH, "n_classes": 3}, tmp_path / "m.pt")
    E.export(str(tmp_path / "m.pt"), str(tmp_path / "m.onnx"), quantize=False)
    rng = np.random.default_rng(0)
    feats = {k: rng.normal(size=(20, 356)).astype(np.float16) for k in ("a", "b", "c")}
    index = [{"id": "a", "gloss": "x y", "splits": {"SI": "dev"}},
             {"id": "b", "gloss": "y", "splits": {"SI": "dev"}},
             {"id": "c", "gloss": "x", "splits": {"SI": "test"}}]
    vocab = {"glosses": ["x", "y"], "lookup": {}}
    res = V.evaluate(str(tmp_path / "m.onnx"), feats, index, vocab, "SI", "dev")
    assert res["n"] == 2 and 0 <= res["wer"] and 0 <= res["sentence_acc"] <= 1
