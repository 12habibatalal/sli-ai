import base64
import io
import json
import tarfile

from training.sentences import kaggle_run as K


def test_bundle_contains_package_not_cache():
    names = tarfile.open(fileobj=io.BytesIO(base64.b64decode(K.bundle())), mode="r:gz").getnames()
    assert "training/sentences/feats.py" in names
    assert not any("/cache" in n or "__pycache__" in n for n in names)


def test_metadata(tmp_path):
    K.write_kernel(tmp_path, "hello", "training.sentences.hello", gpu=True, datasets=["a/b"], kernels=[], args="--x 1")
    meta = json.load(open(tmp_path / "kernel-metadata.json"))
    assert meta["id"] == "baraasaad/sli-sent-hello" and meta["enable_gpu"] is True
    assert meta["dataset_sources"] == ["a/b"] and meta["is_private"] is True
    run = (tmp_path / "run.py").read_text()
    assert "training.sentences.hello" in run and "--x 1" in run


def test_unpacks_to_a_writable_dir(tmp_path):
    # /kaggle/src is read-only on Kaggle (hello kernel v1 failed there)
    K.write_kernel(tmp_path, "hello", "training.sentences.hello", gpu=False, datasets=[], kernels=[])
    run = (tmp_path / "run.py").read_text()
    assert '"/kaggle/src"' not in run and '"/tmp/sli"' in run


def test_other_account(tmp_path):
    K.write_kernel(tmp_path, "r2", "training.sentences.train", gpu=True, datasets=[], kernels=[], user="selia097")
    assert json.load(open(tmp_path / "kernel-metadata.json"))["id"] == "selia097/sli-sent-r2"
    env = K.account_env(2)
    assert env["KAGGLE_API_TOKEN"].startswith("KGAT_") and K.ACCOUNTS[2][0] == "selia097"


def test_bundles_wheels_for_offline_install(tmp_path, monkeypatch):
    # teammate accounts may have no internet in kernels: pose-format must install offline
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    (wheels / "pose_format-0.15.0-py3-none-any.whl").write_bytes(b"x")
    monkeypatch.setattr(K, "WHEELS", str(wheels))
    names = tarfile.open(fileobj=io.BytesIO(base64.b64decode(K.bundle())), mode="r:gz").getnames()
    assert "wheels/pose_format-0.15.0-py3-none-any.whl" in names
    K.write_kernel(tmp_path / "k", "x", "training.sentences.hello", gpu=False, datasets=[], kernels=[])
    assert "--no-index" in (tmp_path / "k" / "run.py").read_text()


def test_find_input_honours_sli_input(tmp_path, monkeypatch):
    (tmp_path / "isharah" / "SI").mkdir(parents=True)
    (tmp_path / "isharah" / "SI" / "train.txt").write_text("id|gloss|text\n")
    monkeypatch.setenv("SLI_INPUT", str(tmp_path))
    assert K.find_input("SI/train.txt") == str(tmp_path / "isharah" / "SI" / "train.txt")
