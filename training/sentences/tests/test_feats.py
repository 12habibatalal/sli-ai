import numpy as np

from training.sentences import feats as F

rng = np.random.default_rng(0)


def fake_raw(with_pose=True, with_face=True, hands=("Left", "Right")):
    pose = rng.random((33, 4)).tolist() if with_pose else None
    face = rng.random((478, 3)).tolist() if with_face else None
    hs = [{"label": lab, "lms": rng.random((21, 3)).tolist()} for lab in hands]
    return {"pose": pose, "face": face, "hands": hs}


def test_layout_and_presence():
    f = F.from_app_raw(fake_raw())
    assert f.shape == (F.FEAT_DIM,) and F.FEAT_DIM == 356
    assert list(f[352:]) == [1, 1, 1, 1]
    empty = F.from_app_raw({"pose": None, "face": None, "hands": []})
    assert not empty.any()


def test_pose_anchor_and_scale():
    raw = fake_raw()
    f = F.from_app_raw(raw)
    p = np.array(raw["pose"])
    w = np.hypot(*(p[11, :2] - p[12, :2]))
    assert np.allclose(f[0:2], 0)  # left shoulder is the origin
    assert np.allclose(f[2:4], (p[12, :2] - p[11, :2]) / w, atol=1e-6)


def test_hands_go_to_nearest_wrist_not_label():
    raw = fake_raw()
    pose = np.array(raw["pose"])
    # the hand labelled "Left" sits at the signer's RIGHT wrist (pose 16)
    lh = np.array(raw["hands"][0]["lms"])
    lh[:, :2] += pose[16, :2] - lh[0, :2]
    rh = np.array(raw["hands"][1]["lms"])
    rh[:, :2] += pose[15, :2] - rh[0, :2]
    raw["hands"][0]["lms"], raw["hands"][1]["lms"] = lh.tolist(), rh.tolist()
    f = F.from_app_raw(raw)
    scale = np.hypot(*(lh[2, :2] - lh[17, :2]))
    assert np.allclose(f[310:352].reshape(21, 2), (lh[:, :2] - lh[0, :2]) / scale, atol=1e-6)


def test_holistic_matches_app_raw():
    raw = fake_raw()
    pose = np.array(raw["pose"])
    for h, wrist in zip(raw["hands"], (15, 16)):  # nearest wrist agrees with Holistic's slots
        lms = np.array(h["lms"])
        lms[:, :2] += pose[wrist, :2] - lms[0, :2]
        h["lms"] = lms.tolist()
    xyz = np.zeros((543, 3))
    conf = np.zeros(543)
    xyz[:33] = pose[:, :3] * [1000, 1000, 1]
    xyz[33:501] = np.array(raw["face"])[:468] * [1000, 1000, 1]
    xyz[501:522] = np.array(raw["hands"][0]["lms"]) * [1000, 1000, 1]
    xyz[522:543] = np.array(raw["hands"][1]["lms"]) * [1000, 1000, 1]
    conf[:] = 1
    assert np.allclose(F.from_holistic(xyz, conf, 1000, 1000), F.from_app_raw(raw), atol=1e-5)


def test_karsl184_face_rescale_is_exact():
    from training import sli_kps

    raw = fake_raw(hands=())
    a = F.from_karsl184(sli_kps.features(raw))
    b = F.from_app_raw(raw)
    assert np.allclose(a[:268], b[:268], atol=1e-5)


def test_resample_irregular_timestamps():
    t = np.array([0, 90, 210, 260, 400, 530], dtype=float)  # ~7-11 fps with jitter
    x = np.stack([np.full(F.FEAT_DIM, v) for v in t])  # value == time, so interpolation is exact
    out = F.resample(x, t)
    assert out.shape[0] == int(530 / (1000 / 15)) + 1
    assert np.allclose(out[:, 0], np.arange(out.shape[0]) * 1000 / 15, atol=1e-3)


def test_mirror_twice_is_identity_and_swaps_hands():
    f = np.stack([F.from_app_raw(fake_raw()) for _ in range(3)])
    m = F.mirror(f)
    assert np.allclose(F.mirror(m), f, atol=1e-5)
    assert np.allclose(m[:, 268:310:2], -f[:, 310:352:2])  # new left x = -(old right x)
    assert np.allclose(m[:, 352:], f[:, [352, 353, 355, 354]])
