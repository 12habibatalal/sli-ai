#!/bin/sh
# (Re)build and start the sentence API; pricelens-proxy reaches it on 127.0.0.1:3021.
set -e
cd "$(dirname "$0")"
mkdir -p ~/sli-models
cp ~/sli-work/models/sentences-large.onnx ~/sli-ai/public/models/sentences-vocab.json ~/sli-models/
docker build -t localhost/sli-sentence-api .
docker rm -f sli-sentence-api 2>/dev/null || true
# No --cpus: rootless podman here has no cpu cgroup controller (only memory, pids). app.py runs one
# inference at a time on one thread, so it never uses more than one core.
docker run -d --name sli-sentence-api --restart=always --memory=1g \
  -p 127.0.0.1:3021:8000 -v ~/sli-models:/models:ro,Z localhost/sli-sentence-api
sleep 15  # the large model takes ~10 s to load
curl -fsS http://127.0.0.1:3021/api/sentence/health
