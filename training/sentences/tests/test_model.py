import numpy as np
import pytest
import torch

from training.sentences import ctc
from training.sentences import model as M


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
    lp, ol = m(x, torch.tensor([41, 20]))
    assert lp.shape == (2, 21, 10) and ol.tolist() == [21, 10]
    alone, _ = m(x[1:, :20], torch.tensor([20]))
    assert torch.allclose(lp[1, :10], alone[0], atol=1e-4)  # padding must not leak


def test_small_is_under_20mb():
    n = sum(p.numel() for p in M.build("small", 700).parameters())
    assert n * 4 < 20e6
