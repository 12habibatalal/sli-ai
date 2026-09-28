# Sentence model: continuous Saudi Sign Language recognition

Date: 2026-09-28. Status: approved in conversation, awaiting spec review.

## Goal

Recognise continuous signing (whole sentences signed without pauses) from people the model has
never seen, with two models, and publish both on Hugging Face:

- **Large** (server): no size limit; runs on the pricelens server behind an API. Most accurate.
- **Small** (phone): ONNX ≤ 20 MB, downloaded by the app and run in the browser, offline.

"Works on new videos" is defined and measured as: word error rate (WER) on signers held out of
training. The model recognises the ~685 glosses of Isharah-1000 (plus KArSL signs used for
pre-training); it cannot recognise signs it was never trained on, and it targets Saudi Sign
Language.

## Data

| Dataset | What | Use | Licence |
| --- | --- | --- | --- |
| Isharah-1000 pose (`tfmohamedyahia/isharah-1000-pose`, Kaggle) | 15,000 continuous videos, 18 signers, 1,000 sentences, 685 glosses; MediaPipe Holistic landmarks (33 pose + 468 face + 2×21 hands), `.pose` format, 25 fps | Main training and evaluation | CC-BY-NC-SA-4.0 |
| Isharah annotations (`abedelmortadhajemai/annotations-isharah`) | `id\|gloss\|text` for SI (signer-independent: 10 train / 1 dev / 4 test signers) and US (unseen sentences) splits | Labels and splits | from Isharah |
| KArSL-502 landmarks (`youssefelkilany/word-level-arabic-sign-language-extrcted-keypoints`) | Isolated signs, 3 signers, keypoints from the original model's author | Stage-1 pre-training | KArSL terms (research) |
| ArabSign (`osamaalmolike/ssl-data-set`, videos) | 50 sentences, 6 signers | External check only (landmarks extracted on Kaggle) | ArabSign terms |

The SI dev signer is used for model selection; the SI test signers are evaluated once, at the end.

## Features

Same landmark subset and normalisation as the app (`training/sli_kps.py`,
`src/recognition/features.ts`): hands relative to their wrists, arms relative to the shoulders
and scaled by shoulder width, the 136 face points in `training/face_idx.json`. Holistic's
468-point face mesh is the first 468 points of the app's FaceLandmarker (478), and the pose and
hand topologies are identical, so the features are the same. Frames are resampled to the app's
frame rate. A parity fixture proves the TypeScript features match Python on Isharah samples.

## Models

Both are landmark models with the same features, vocabulary and CTC output.

### Large (server)

- Deeper/wider Transformer encoder with CTC head; size chosen by dev WER, not by file size.
- Hard limit is latency, not size: one sentence (≈20 s of signing) must decode in ≤ 2 s on one
  core of the pricelens server (2× Neoverse-N1, no GPU), so it does not slow the other sites.
  Checked with ONNX Runtime on the server before release; if it is too slow, the next smaller
  size that meets the limit is used.

### Small (phone)

- Distilled from the large model (trained on its outputs plus the true labels).

### Both

- Transformer encoder over per-frame features (temporal downsampling ×2), CTC head over the
  gloss vocabulary + blank.
- Stage 1: pre-train the encoder on KArSL isolated signs (each clip = a one-gloss sequence).
- Stage 2: train on Isharah SI train.
- Augmentation aimed at unseen signers: body scale/aspect, small rotations, left/right mirroring
  (with hand swap), speed change 0.7–1.3×, random landmark/hand dropout, jitter.
- Selection: lowest WER on SI dev; greedy CTC decoding (beam search tried, kept only if it helps).
- Small model budget: ONNX file ≤ 20 MB (int8 quantisation if needed), fast enough for a phone browser.

## Training infrastructure

Kaggle GPU notebooks under the user's account (`baraasaad`), pushed and polled from the
pricelens server with the Kaggle CLI (`~/sli-work/kvenv`, Python 3.12, token in
`~/.kaggle/access_token`). Preprocessing runs once and is saved as a private Kaggle dataset.
Nothing heavy runs on the pricelens server. Code lives in `training/sentences/` in this repo;
notebooks are generated from it so the repo stays the source of truth.

## Stopping rule

Train in rounds; each round changes one thing and is compared on SI dev WER. Stop when the
model matches or beats the Isharah paper's pose-based SI baseline (its best SI baseline of any
kind if it reports no pose-based one), or when two consecutive
rounds stop improving dev WER, whichever comes first — then run the final test once.

## Evaluation (reported in `src/data/metrics.json`, README and model card)

Every number is reported for both models.

- WER on SI test (4 unseen signers) — headline number.
- WER on US test (unseen sentences).
- WER / sentence accuracy on ArabSign (different dataset, different people).
- Browser replay: the web implementation on sample Isharah videos matches Python's output.

## Gloss → text

A lookup from gloss sequences to Isharah's written sentences: an exact match shows the
sentence, otherwise the glosses are shown joined. A translation model is out of scope.

## Server API (large model)

- Small Python service (ONNX Runtime) in a container on pricelens, published on a loopback port
  and reached through the shared `pricelens-proxy` with a vhost under `docker/nginx-upstreams`
  that points at `127.0.0.1:<port>` (never a container name), e.g. `/api/sentence` on the SLI
  site.
- Input: the per-frame features for one sentence (a few KB). No video or images ever leave the
  phone.
- Output: glosses, text, per-gloss confidence.
- Limits: request size cap, one inference at a time per core (queue), per-IP rate limit,
  timeout; CPU capped so the other sites keep working. Stateless; nothing is stored.

## App

New "Sentences" mode: rolling window of frames → features (existing MediaPipe worker) → large
model through the API when online; small model in the recognition worker when offline or when
the API is slow or unavailable. CTC decode → glosses and text on screen. Reuses the framing
coach. Tests: unit (features parity, CTC decoding), replay (vitest) for both models, API tests
(latency, limits, fallback), e2e (Playwright, as for words).

## Hugging Face release

Model repo under the user's account with both models (large and small ONNX), feature spec, gloss vocabulary, lookup table,
metrics, model card (intended use, limits: vocabulary, Saudi Sign Language, unseen-signer WER),
licence CC-BY-NC-SA-4.0, citations for Isharah, KArSL, the MIT ST-Transformer and pose-format.
Only weights and metadata are published, never videos or landmarks. Requires a Hugging Face
write token from the user at release time.

## Out of scope

RGB/video models, signs outside Isharah/KArSL, other sign languages, gloss-to-text translation
models, commercial use.
