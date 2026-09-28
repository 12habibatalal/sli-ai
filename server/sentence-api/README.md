# server/sentence-api

The large sentence model behind `POST /api/sentence` (see `app.py` for the request format).

- Deploy: `./run.sh`. It copies `~/sli-work/models/sentences-large.onnx` and
  `public/models/sentences-vocab.json` to `~/sli-models`, builds the image and starts the
  container `sli-sentence-api` (restart=always, 1 GB memory, `127.0.0.1:3021`).
- CPU: rootless podman on the server has no cpu cgroup controller, so there is no `--cpus`.
  `app.py` runs one inference at a time on one thread (a second request waits up to 4 s, then
  gets 503).
- Public route: `location /api/sentence` in `~/pricelens/docker/nginx-upstreams/sli.conf`
  (untracked in the pricelens repo). It rate-limits to 20 requests a minute per IP with a burst
  of 10 (then 429), caps bodies at 700 KB, and proxies to `http://127.0.0.1:3021` (the proxy
  runs in the host network, so never use a container name). Reload the proxy with
  `docker exec pricelens-proxy nginx -t && docker exec pricelens-proxy nginx -s reload`.
- Nothing is logged or stored: only SF1 numbers arrive, never video.
- Tests: `python -m pytest` in this folder (needs numpy, onnx, onnxruntime).
