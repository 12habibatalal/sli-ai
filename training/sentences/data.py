"""Training data: SF1 sequences, augmentation aimed at unseen signers and cameras, batching."""
import numpy as np
import torch

from training.sentences import feats as F

PARTS = [(F.O_POSE, 6, 0), (F.O_FACE, 128, 1), (F.O_LH, 21, 2), (F.O_RH, 21, 3)]  # offset, points, flag


def _affine(pts, rng, rot, scale):
    a = np.deg2rad(rng.uniform(-rot, rot))
    sx, sy = rng.uniform(1 - scale, 1 + scale, 2)
    m = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) @ np.diag([sx, sy])
    return pts @ m.T


# Augmentation levels. Round 1 memorised its 10 training signers; strong (round 3) cut dev WER
# from 26.2 to 21.5, xstrong (round 5) pushes further along the same axis.
AUG = {
    "normal": dict(rot=(15, 10), scale=(0.2, 0.15), speed=(0.7, 1.3), drop_p=0.3, drop_len=8, drops=1),
    "strong": dict(rot=(25, 15), scale=(0.3, 0.2), speed=(0.6, 1.4), drop_p=0.5, drop_len=16, drops=2),
    "xstrong": dict(rot=(35, 20), scale=(0.4, 0.25), speed=(0.5, 1.5), drop_p=0.6, drop_len=22, drops=3),
    "xxstrong": dict(rot=(45, 25), scale=(0.5, 0.3), speed=(0.45, 1.6), drop_p=0.7, drop_len=26, drops=4),
}


def augment(x, rng, level="normal"):
    a = AUG[level]
    x = x.astype(np.float32, copy=True)
    if rng.random() < 0.5:  # left-handed signers
        x = F.mirror(x)
    T = len(x)
    for o, n, k in PARTS:  # body proportions, camera angle and aspect ratio, landmark noise
        pts = x[:, o:o + 2 * n].reshape(T, n, 2)
        i = 0 if k == 0 else 1
        pts = _affine(pts, rng, a["rot"][i], a["scale"][i]) + rng.normal(0, 0.01, pts.shape)
        x[:, o:o + 2 * n] = pts.reshape(T, -1) * x[:, [F.O_PRES + k]]
    speed = rng.uniform(*a["speed"])  # signing speed
    x = F.resample(x, np.arange(T) * speed * 1000.0 / F.FPS)
    T = len(x)
    for o, n, k in PARTS[2:]:  # hand tracking losses (fast motion, one hand behind the other)
        for _ in range(a["drops"]):
            if rng.random() < a["drop_p"]:
                s = rng.integers(0, T)
                e = min(T, s + rng.integers(1, a["drop_len"]))
                x[s:e, o:o + 2 * n] = 0
                x[s:e, F.O_PRES + k] = 0
    if rng.random() < 0.05:  # face not found at all
        x[:, F.O_FACE:F.O_FACE + 256] = 0
        x[:, F.O_PRES + 1] = 0
    return x


def stress(x, kind, rng):
    """Harder versions of a test clip, for evaluation only.

    hand: one hand at a time disappears for 0.5-1.5 s stretches covering about a third of the
          clip, as when a hand passes behind the other (the arms stay tracked);
    fast: the same signing 1.5 times faster.
    """
    x = np.array(x, np.float32, copy=True)
    T = len(x)
    if kind == "fast":
        return F.resample(x, np.arange(T) * (1000.0 / F.FPS) / 1.5)
    if kind != "hand":
        raise ValueError(kind)
    hidden = np.zeros(T, bool)
    for _ in range(200):
        room = int(0.4 * T) - hidden.sum()
        if hidden.sum() >= 0.3 * T or room < 8:
            break
        n = min(int(rng.integers(8, 23)), room)
        s = int(rng.integers(0, max(T - n, 0) + 1))
        if hidden[max(s - 1, 0):s + n + 1].any():  # keep stretches apart: one hand at a time
            continue
        o, k = (F.O_LH, 2) if rng.random() < 0.5 else (F.O_RH, 3)
        x[s:s + n, o:o + 42] = 0
        x[s:s + n, F.O_PRES + k] = 0
        hidden[s:s + n] = True
    return x


class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, feats, items, train, seed=0, level="normal"):
        self.feats, self.items, self.train, self.level = feats, items, train, level
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        key, ids = self.items[i]
        x = np.asarray(self.feats[key], np.float32)
        if self.train:
            x = augment(x, self.rng, self.level)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.tensor(ids, dtype=torch.long)


def collate(batch):
    xs, ys = zip(*batch)
    xl = torch.tensor([len(x) for x in xs])
    x = torch.zeros(len(xs), int(xl.max()), F.FEAT_DIM)
    for i, v in enumerate(xs):
        x[i, : len(v)] = v
    return x, xl, torch.cat(ys), torch.tensor([len(y) for y in ys])


def bucket_batches(lengths, max_frames, rng):
    """Batches of similar length whose padded size (longest x count) stays within max_frames."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches, cur, longest = [], [], 0
    for i in order:
        if cur and max(longest, lengths[i]) * (len(cur) + 1) > max_frames:
            batches.append(cur)
            cur, longest = [], 0
        cur.append(i)
        longest = max(longest, lengths[i])
    if cur:
        batches.append(cur)
    rng.shuffle(batches)
    return batches
