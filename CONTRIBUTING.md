# Contributing to SLI

Thanks for helping. SLI is useful only if it understands real signers, so better models, bug
reports from Deaf signers and new data matter as much as code.

- **Questions and ideas:** start a [discussion](https://github.com/Digital-Fingers-Team/sli-ai/discussions).
- **Bugs:** open an issue. For recognition errors, say which mode (Auto, Words, Letters,
  Sentences), what you signed and what came out, and your device and browser.
- **Code:** open a pull request against `main`. Keep it to one change and say how you tested it.
- **Models:** see [Improving the sentence models](#improving-the-sentence-models) below.

## Setup

App (Node 20+):

```bash
npm install
npm run dev          # http://localhost:5173
npm test             # unit tests
npm run build        # typecheck + production build
```

Training (Python 3.10+):

```bash
pip install torch numpy onnx onnxruntime pose-format pytest
python -m pytest training/sentences/tests
```

The end-to-end browser tests (`npx playwright test`) need camera fixtures built from KArSL videos.
See `README.md` → Develop.

## Improving the sentence models

The sentence models read Saudi Sign Language sentences with CTC. They were trained on
[Isharah-1000](https://www.kaggle.com/datasets/tfmohamedyahia/isharah-1000-pose), after
pre-training on KArSL. The released weights, ONNX and PyTorch, are on Hugging Face:
[DF-Team/sli-sentences](https://huggingface.co/DF-Team/sli-sentences).

### Where the models stand

Word error rate on the Isharah signer-independent (SI) **dev** split:

| Model | Dev WER |
| --- | --- |
| SLI Sentences Large v2 | 17.5% |
| SLI Sentences Phone v1 | 17.4% |
| Best pose-based entry, MSLR 2025 challenge (arXiv:2508.09372) | 7.3% |

There is plenty of room. Ideas not tried yet: the US (unseen sentences) protocol, signer
normalisation, better use of the face, a language model over glosses, and sentence-level data for
other Arabic sign languages. `training/sentences/rounds.md` lists what was tried, including what
did not help (higher dropout, heavier augmentation than `xstrong`).

### Data

All three datasets are public on Kaggle:

| Dataset | Kaggle slug |
| --- | --- |
| Isharah-1000 pose files | `tfmohamedyahia/isharah-1000-pose` |
| Isharah annotations (SI / US splits) | `abedelmortadhajemai/annotations-isharah` |
| KArSL keypoints (pre-training) | `youssefelkilany/word-level-arabic-sign-language-extrcted-keypoints` |

Isharah is licensed CC BY-NC-SA 4.0, so every model trained on it is too. Check each dataset's
terms before using it.

### Train

The code finds its inputs under `$SLI_INPUT` (default `/kaggle/input`) and writes to `$SLI_OUT`.

**On Kaggle (free GPU):** create a notebook, attach the three datasets, enable the GPU, clone this
repo and run the commands below with `SLI_OUT=/kaggle/working`. A large run takes 2–3 GPU hours.
Maintainers can also launch kernels from a terminal with `training/sentences/kaggle_run.py`
(set `SLI_KAGGLE_USER` to your username).

**Locally:** download the datasets into one folder and point `SLI_INPUT` at it.

```bash
export SLI_INPUT=~/data SLI_OUT=~/runs

# 1. Landmarks -> SF1 features at 15 fps, plus the vocabulary (CPU, once)
python -m training.sentences.prep

# 2. Pre-train on KArSL isolated signs, then train on Isharah SI
python -m training.sentences.train --stage karsl --model large --epochs 30
python -m training.sentences.train --stage isharah --model large --epochs 100 --aug xstrong \
    --init $SLI_OUT/karsl-large.pt
# (or skip the first command: --init karsl-pretrain-large.pt, from the Hugging Face repo's pytorch/ folder)

# Or start from the released weights. --keep-head keeps their trained classifier head.
python -m training.sentences.train --stage isharah --model large --epochs 30 --aug xstrong \
    --init pytorch/sli-sentences-large-v2.pt --keep-head

# Distil a large teacher into the phone-sized model
python -m training.sentences.train --stage distill --model small --epochs 80 --aug xstrong \
    --teacher $SLI_OUT/isharah-large.pt
```

### Evaluate

Always export first and score the ONNX file. That is what the app runs.

```bash
python -m training.sentences.export $SLI_OUT/isharah-large.pt large.onnx
python -m training.sentences.evaluate large.onnx --stress hand fast noface
```

Rules for comparing results:

- Choose models and settings on **SI dev** only. Evaluate on SI **test** once, at the end, and
  report both numbers.
- Report the stress tests (one hand hidden, 1.5× faster, no face) next to plain WER.
- Change one thing per round and add a row to `training/sentences/rounds.md`, including runs
  that did not help.

### Submit a model

1. Open a pull request here with the code change and your `rounds.md` row: the command line,
   dev WER, stress tests and GPU hours.
2. Put the weights (`.pt` and exported `.onnx`) somewhere public, such as a Hugging Face model
   repo or a pull request on [DF-Team/sli-sentences](https://huggingface.co/DF-Team/sli-sentences/discussions).
3. We re-run `evaluate.py` on your ONNX. If it beats the current model on dev without getting
   worse on the stress tests, it becomes the next release and you are credited in the model card.

The SF1 feature layout (`training/sentences/feats.py`, ported in
`src/recognition/sentenceFeatures.ts`) and the vocabulary are shared with the app. A change to
either needs the browser port and its parity test (`tests/unit/sentenceFeatures.test.ts`) updated
in the same pull request.

## Licences

- Code: MIT (`LICENSE`).
- Sentence models (`public/models/sentences-*`, Hugging Face): CC BY-NC-SA 4.0, from Isharah.
- Word model `public/models/karsl502.onnx`: MIT, by Yousef Elkilany. See `README.md` → Credits.
- By contributing, you agree to license your contribution under the same terms.
