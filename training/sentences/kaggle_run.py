"""Runs a module of this repo as a private Kaggle kernel (GPU optional), waits, pulls outputs.

    python -m training.sentences.kaggle_run NAME --entry training.sentences.X [--gpu]
        [--datasets owner/slug ...] [--kernels owner/slug ...] [--args "..."] [--wait] [--pull DIR]

The kernel unpacks training/ from an embedded tarball, pip-installs PIP, and runs
`python -m ENTRY ARGS` with SLI_OUT=/kaggle/working (the kernel's output directory).
"""
import argparse
import base64
import glob
import io
import json
import os
import subprocess
import tarfile
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
KAGGLE = os.path.expanduser("~/sli-work/kvenv/bin/kaggle")
USER = "baraasaad"
# --account N: (username, token file). 2 and 3 are teammates' accounts lent to the project.
ACCOUNTS = {
    1: (USER, "~/.kaggle/access_token"),
    2: ("selia097", "~/.kaggle-t2/access_token"),
    3: ("mono768", "~/.kaggle-t3/access_token"),
}


def account_env(n):
    """Environment for the kaggle CLI acting as account n."""
    env = dict(os.environ)
    env["KAGGLE_API_TOKEN"] = open(os.path.expanduser(ACCOUNTS[n][1])).read().strip()
    return env
PIP = "pose-format onnx onnxruntime"

# /kaggle/src (where the script lives) is read-only, so the code is unpacked into /tmp/sli.
RUN = '''import base64, io, os, subprocess, sys, tarfile
tarfile.open(fileobj=io.BytesIO(base64.b64decode(BUNDLE)), mode="r:gz").extractall("/tmp/sli", filter="data")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", *"{pip}".split()], check=False)
os.environ["SLI_OUT"] = "/kaggle/working"
os.chdir("/tmp/sli")
subprocess.run(["find", "/kaggle/input", "-maxdepth", "4", "-type", "d"], check=False)
sys.exit(subprocess.run([sys.executable, "-m", "{entry}", *{args!r}.split()]).returncode)
'''


def bundle():
    buf = io.BytesIO()

    def skip(ti):
        n = ti.name
        return None if ("/cache" in n or "__pycache__" in n or ".pytest_cache" in n or n.endswith(".log")) else ti

    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(os.path.join(ROOT, "training"), arcname="training", filter=skip)
    return base64.b64encode(buf.getvalue()).decode()


def write_kernel(d, name, entry, gpu, datasets, kernels, args="", user=USER):
    os.makedirs(d, exist_ok=True)
    meta = {"id": f"{user}/sli-sent-{name}", "title": f"sli-sent-{name}", "code_file": "run.py",
            "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": gpu,
            "enable_internet": True, "dataset_sources": datasets, "kernel_sources": kernels,
            "competition_sources": []}
    json.dump(meta, open(os.path.join(d, "kernel-metadata.json"), "w"), indent=1)
    with open(os.path.join(d, "run.py"), "w") as f:
        f.write(f"BUNDLE = {bundle()!r}\n" + RUN.format(pip=PIP, entry=entry, args=args))


def find_input(pattern):
    """First match of pattern under /kaggle/input (inside a kernel)."""
    hits = sorted(glob.glob(os.path.join("/kaggle/input", "**", pattern), recursive=True))
    if not hits:
        raise FileNotFoundError(pattern)
    return hits[0]


def status(slug, env=None):
    out = subprocess.run([KAGGLE, "kernels", "status", slug], capture_output=True, text=True, env=env)
    text = (out.stdout + out.stderr).lower()
    for s in ("complete", "error", "cancel", "running", "queued"):
        if s in text:
            return s
    return text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--entry", required=True)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--datasets", nargs="*", default=[])
    ap.add_argument("--kernels", nargs="*", default=[])
    ap.add_argument("--args", default="")
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--pull")
    ap.add_argument("--pattern", default=r"(\.json|\.log|\.pt|\.onnx)$", help="output files to pull")
    ap.add_argument("--account", type=int, default=1, choices=sorted(ACCOUNTS))
    a = ap.parse_args()
    user, env = ACCOUNTS[a.account][0], account_env(a.account)
    d = os.path.expanduser(f"~/sli-work/kaggle/{user}/{a.name}")
    write_kernel(d, a.name, a.entry, a.gpu, a.datasets, a.kernels, a.args, user=user)
    subprocess.run([KAGGLE, "kernels", "push", "-p", d], check=True, env=env)
    slug = f"{user}/sli-sent-{a.name}"
    while a.wait:
        time.sleep(60)
        s = status(slug, env)
        print(time.strftime("%H:%M"), s, flush=True)
        if s in ("complete", "error", "cancel"):
            break
    if a.pull:
        os.makedirs(a.pull, exist_ok=True)
        subprocess.run([KAGGLE, "kernels", "output", slug, "-p", a.pull, "-o", "--file-pattern", a.pattern],
                       check=False, env=env)


if __name__ == "__main__":
    main()
