import numpy as np

from training.sentences import evaluate as V
from training.sentences import feats as F


def clip(T=150):
    x = np.random.default_rng(0).normal(0, 0.3, (T, F.FEAT_DIM)).astype(np.float32)
    x[:, F.O_PRES:] = 1
    return x


def test_hidden_hand_removes_about_a_third_of_one_hands_frames_in_long_stretches():
    x = clip()
    y = V.stress(x, "hand", np.random.default_rng(1))
    gone_l = y[:, F.O_PRES + 2] == 0
    gone_r = y[:, F.O_PRES + 3] == 0
    assert not (gone_l & gone_r).any()  # one hand at a time, as when it passes behind the other
    frac = (gone_l | gone_r).mean()
    assert 0.2 <= frac <= 0.45
    assert not y[gone_l, F.O_LH:F.O_LH + 42].any() and not y[gone_r, F.O_RH:F.O_RH + 42].any()
    runs = np.diff(np.flatnonzero(np.diff(np.r_[0, (gone_l | gone_r).astype(int), 0])))[::2]
    assert runs.min() >= 8  # stretches of 0.5 s or more at 15 fps
    assert np.array_equal(y[:, F.O_POSE:F.O_POSE + 12], x[:, F.O_POSE:F.O_POSE + 12])  # arms still tracked


def test_fast_is_one_and_a_half_times_shorter():
    y = V.stress(clip(150), "fast", np.random.default_rng(0))
    assert len(y) == 100


def test_noface_removes_the_whole_face_only():
    x = clip(60)
    y = V.stress(x, "noface", np.random.default_rng(0))
    assert not y[:, F.O_FACE:F.O_FACE + 256].any() and not y[:, F.O_PRES + 1].any()
    assert np.array_equal(np.delete(y, np.r_[F.O_FACE:F.O_FACE + 256, F.O_PRES + 1], axis=1),
                          np.delete(x, np.r_[F.O_FACE:F.O_FACE + 256, F.O_PRES + 1], axis=1))
