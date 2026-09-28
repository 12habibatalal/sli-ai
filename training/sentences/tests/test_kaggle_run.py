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
