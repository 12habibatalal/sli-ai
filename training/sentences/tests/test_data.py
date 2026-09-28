import numpy as np

from training.sentences import data as D
from training.sentences import feats as F

rng = np.random.default_rng(0)


def sample(T=60):
    x = rng.normal(0, 0.3, (T, F.FEAT_DIM)).astype(np.float32)
    x[:, F.O_PRES:] = 1
    return x


def test_augment_keeps_layout_and_presence_semantics():
    for _ in range(50):
        y = D.augment(sample(), rng)
        assert y.shape[1] == F.FEAT_DIM and 30 <= y.shape[0] <= 100
        assert np.isfinite(y).all()
        for part, (o, n) in {2: (F.O_LH, 42), 3: (F.O_RH, 42)}.items():
            gone = y[:, F.O_PRES + part] == 0
            assert not y[gone, o:o + n].any()  # a dropped hand is all zeros with flag 0


def test_collate_and_buckets():
    ds = D.SeqDataset({"a": sample(40), "b": sample(25)}, [("a", [1, 2]), ("b", [3])], train=False)
    x, xl, y, yl = D.collate([ds[0], ds[1]])
    assert tuple(x.shape) == (2, 40, 356) and xl.tolist() == [40, 25]
    assert y.tolist() == [1, 2, 3] and yl.tolist() == [2, 1]
    lengths = [10, 300, 20, 310, 15]
    b = D.bucket_batches(lengths, max_frames=640, rng=rng)
    assert sorted(i for bb in b for i in bb) == [0, 1, 2, 3, 4]
    assert all(max(lengths[i] for i in bb) * len(bb) <= 640 for bb in b)


def test_strong_augment_keeps_invariants_and_varies_more():
    lens_n, lens_s, gone_n, gone_s = [], [], 0, 0
    r1, r2 = np.random.default_rng(3), np.random.default_rng(3)
    for _ in range(200):
        n = D.augment(sample(), r1)
        s = D.augment(sample(), r2, level="strong")
        assert np.isfinite(s).all() and s.shape[1] == F.FEAT_DIM
        for o, k in ((F.O_LH, 2), (F.O_RH, 3)):
            assert not s[s[:, F.O_PRES + k] == 0, o:o + 42].any()
        lens_n.append(len(n)); lens_s.append(len(s))
        gone_n += int((n[:, F.O_PRES + 2:F.O_PRES + 4] == 0).sum())
        gone_s += int((s[:, F.O_PRES + 2:F.O_PRES + 4] == 0).sum())
    assert np.ptp(lens_s) > np.ptp(lens_n)  # wider speed range
    assert gone_s > 1.5 * gone_n  # more hidden-hand frames


def test_xstrong_varies_more_than_strong():
    lens_s, lens_x, gone_s, gone_x = [], [], 0, 0
    r1, r2 = np.random.default_rng(5), np.random.default_rng(5)
    for _ in range(200):
        s = D.augment(sample(), r1, level="strong")
        x = D.augment(sample(), r2, level="xstrong")
        assert np.isfinite(x).all() and x.shape[1] == F.FEAT_DIM
        lens_s.append(len(s)); lens_x.append(len(x))
        gone_s += int((s[:, F.O_PRES + 2:F.O_PRES + 4] == 0).sum())
        gone_x += int((x[:, F.O_PRES + 2:F.O_PRES + 4] == 0).sum())
    assert np.ptp(lens_x) > np.ptp(lens_s) and gone_x > gone_s
