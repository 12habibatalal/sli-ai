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


def test_attention_matches_torch_multihead_attention():
    # same parameter names and numbers as nn.MultiheadAttention (older checkpoints load)
    torch.manual_seed(0)
    ref = torch.nn.MultiheadAttention(32, 4, batch_first=True).eval()
    att = M.SelfAttention(32, 4, 0.0).eval()
    assert set(att.state_dict()) == set(ref.state_dict())
    att.load_state_dict(ref.state_dict())
    x = torch.randn(2, 9, 32)
    mask = torch.tensor([[False] * 9, [False] * 6 + [True] * 3])
    want = ref(x, x, x, key_padding_mask=mask, need_weights=False)[0]
    assert torch.allclose(att(x, mask), want, atol=1e-5)


def test_build_takes_dropout():
    m = M.build("small", 10, dropout=0.3)
    assert m.blocks[0].drop.p == 0.3 and m.blocks[0].att.dropout == 0.3
