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
