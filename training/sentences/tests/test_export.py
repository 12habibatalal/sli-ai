import numpy as np
import onnxruntime as ort
import torch

from training.sentences import export as E
from training.sentences import model as M

ARCH = dict(d=64, layers=2, heads=4, ff=128, conv=3)


def _ckpt(tmp_path):
    torch.manual_seed(0)
    m = M.SignCTC(12, **ARCH).eval()
    path = tmp_path / "m.pt"
    torch.save({"state": m.state_dict(), "config": None, "arch": ARCH, "n_classes": 12}, path)
    return m, path


def test_onnx_matches_torch_at_any_length(tmp_path):
    m, ck = _ckpt(tmp_path)
    out = tmp_path / "m.onnx"
    E.export(str(ck), str(out), quantize=False)
    s = ort.InferenceSession(str(out))
    for T in (9, 57, 300):
        x = np.random.default_rng(T).normal(size=(1, T, 356)).astype(np.float32)
        ref, _ = m(torch.from_numpy(x), torch.tensor([T]))
        got = s.run(["logprobs"], {"feats": x})[0]
        assert got.shape == (1, (T + 1) // 2, 12)
        assert np.allclose(got, ref.detach().numpy(), atol=1e-4)


def test_int8_is_smaller_and_close(tmp_path):
    m, ck = _ckpt(tmp_path)
    fp, q = tmp_path / "fp.onnx", tmp_path / "q.onnx"
    E.export(str(ck), str(fp), quantize=False)
    E.export(str(ck), str(q), quantize=True)
    assert q.stat().st_size < fp.stat().st_size
    x = np.random.default_rng(1).normal(size=(1, 40, 356)).astype(np.float32)
    a = ort.InferenceSession(str(fp)).run(None, {"feats": x})[0]
    b = ort.InferenceSession(str(q)).run(None, {"feats": x})[0]
    assert np.abs(np.exp(a) - np.exp(b)).max() < 0.1


def test_latency_reports_seconds(tmp_path):
    _, ck = _ckpt(tmp_path)
    out = tmp_path / "m.onnx"
    E.export(str(ck), str(out), quantize=False)
    assert 0 < E.latency(str(out), frames=60) < 5
