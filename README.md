# SLI — AI Arabic Sign Language Interpreter

SLI translates **Arabic Sign Language ⇄ Arabic text ⇄ speech**, entirely in the browser.

- **Sign → text → speech.** Sign in front of the camera; the word appears while you sign and
  builds a sentence that can be spoken aloud in Arabic. Each word can be swapped for the
  model's next-best guesses.
- **Live fingerspelling.** The hand shape and the wrist's last few positions are read every
  frame, so letters are typed one after another without lowering the hand.
- **Teach the app your hand.** Sign each letter once (about two minutes) and the letter model
  is fine-tuned on the device to that person: on signers it never saw, 77% → 86% of letters.
- **Words and letters together.** In Auto mode (the default) a still, raised hand is read as a
  letter and movement as a word, so a sentence can mix signed words and spelled names.
  Words and Letters modes restrict it to one kind.
- **One or two hands.** Both hands are tracked; two-handed signs use both.
- **Framing coach.** If a hand stays outside the picture while its elbow is in view, or the
  signer is too close, the camera view says so (move back, or turn the phone sideways): the
  model learned from signers filmed with room around them.
- **Text or voice → sign.** Type or speak Arabic; each word is shown as a real signer video.
  Unknown words and names are fingerspelled with letter signs, and numbers are composed from
  number signs.
- **Conversation.** A deaf and a hearing person talk on one screen, with a shared transcript.
- **Dictionary.** All 502 signs, searchable, with a video for each.

Recognition runs on the device (MediaPipe + ONNX Runtime Web), in Web Workers so the camera
view stays smooth. Camera images never leave the device, and after the first visit the app
works offline.

## How it works

| Step | What happens | Code |
| --- | --- | --- |
| Landmarks | MediaPipe in a worker: hands detected on every frame, pose and face tracked | `src/recognition/landmarks*.ts` |
| Features | 184 keypoints × (x, y, z, visibility), normalised exactly as in training | `src/recognition/features.ts` |
| Segmenting | Commits a sign when the next one takes over or the hands drop for 300 ms | `src/recognition/decoder.ts` |
| Classifying | Frames resampled to 50 and classified by an ST-Transformer over 502 KArSL signs, in its own worker | `src/recognition/classifier*.ts` |
| Letters | A per-frame hand-shape model (trained here on KArSL's letter videos) and a hold-to-type speller | `src/recognition/letters.ts`, `speller.ts` |
| Auto mode | Runs both; letters only while the hand is still and raised and the word model does not see a word | `src/recognition/interpreter.ts` |
| Text → sign | Arabic normalisation, phrase matching, light stemming, numbers, fingerspelling | `src/translate/` |

The recognition model is the MIT-licensed
[Word-level Arabic Sign Language ST-Transformer](https://github.com/yousefelkilany/Word-level-Arabic-Sign-language),
trained on [KArSL-502](https://hamzah-luqman.github.io/KArSL/) (Unified Arabic Sign Language: 31 numbers,
39 letters and 432 words).

## Sentence models

Sentences mode reads a whole Saudi Sign Language sentence at once (CTC, trained on
[Isharah-1000](https://www.kaggle.com/datasets/tfmohamedyahia/isharah-1000-pose), 672 signs, after pre-training on KArSL).
Word error rate (WER) is measured on the Isharah signer-independent dev split, whose signer is never seen in training:

| Model | Runs | Size | WER | One hand hidden | 1.5× faster | No face |
| --- | --- | --- | --- | --- | --- | --- |
| **SLI Sentences Large v2** (`large-v2`) | server, `POST /api/sentence` | 169 MB, 42M params | 17.5% | 18.0% | 16.8% | 19.2% |
| **SLI Sentences Phone v1** (`sentences-small.onnx`) | in the browser, offline | 17.9 MB | 17.4% | 18.0% | 16.7% | 19.4% |

The app sends each sentence to the server model and falls back to the phone model when the server
is slow (4 s) or unreachable. The phone model learnt from a large teacher (distillation), so the two
score about the same. For comparison, the best signer-independent dev result in the Isharah paper is 17.9% (Swin-MSTP, RGB video),
and the best pose-based entry in the MSLR 2025 challenge reached 7.3%, so there is room to improve.
Training rounds and stress tests: `training/sentences/rounds.md`. Licence of both models: CC-BY-NC-SA-4.0 (from Isharah).
Weights (ONNX and PyTorch): [DF-Team/sli-sentences](https://huggingface.co/DF-Team/sli-sentences) on Hugging Face.

## Measured accuracy

On the KArSL **test** split (details and method in `training/README.md`):

- **Isolated signs:** 97.0% top-1, 99.6% top-5 over 1,506 videos (502 signs × 3 signers).
- **3-sign sentences through the app pipeline, 15 fps:** 93% of words with a short pause
  between signs, 54% when signing straight through without lowering the hands.
- **Letters, on a signer the model never saw:** 77.0% (leave-one-signer-out over KArSL's three
  signers), 86.4% after that person teaches the app their hand (one recording per letter).
  Most errors are letters sharing a hand shape (ي/ى/ئ, ت/ة, ج/ح, ز/ذ).
- **Auto mode, word–letter–letter–word sentences with brief pauses:** 91% (letter model that
  never saw the signer). Spelling without pauses is better in Letters mode (78% vs 43%).
- **Hand location matters:** moving the same movement to another sign's place drops the word
  model from 96.8% to 62.5%, so where the hands are relative to the body is part of every
  word it reads.
- **Two hands:** 99.0% top-1 with both hands given to the model vs 97.0% with one (504 videos).

The word model was trained on sign clips that start and end with the hands down, so it is best
with a brief pause between signs. Its three test signers also appear in its training data, so
accuracy with new people is lower. Good light and the whole upper body in frame help most.

## Develop

```bash
npm install
npm run dev          # http://localhost:5173
npm test             # unit tests (features parity with Python, decoder, text→sign planner)
npm run build        # typecheck + production build in dist/
```

End-to-end tests drive real Chromium with KArSL videos as the camera:

```bash
python training/make_fake_camera.py tests/fixtures/camera.y4m KARSL_DIR 01 290 497 --slow 4
python training/make_fake_camera.py tests/fixtures/camera-letters.y4m KARSL_DIR 01 33 54 --slow 4
python training/make_fake_camera.py tests/fixtures/camera-close.y4m KARSL_DIR 01 289 --slow 4 --crop 0.5,0.67
npx playwright test
```

`?delegate=CPU` in the URL forces MediaPipe onto the CPU (for devices with broken WebGL),
`?handsMode=video` / `?body=image` change the per-part tracking modes, `?hands=1` tracks one hand, `?bodyEvery=N` fixes how
often pose and face are refreshed, and `?debug` keeps raw landmarks on `window.__sliFrames`.

## Contributing

Issues and pull requests are welcome, especially better sentence models. [CONTRIBUTING.md](CONTRIBUTING.md) covers
the data, the training commands and how models are compared.

## Credits

- **Digital Fingers team** — Anas Mohamed Mokhtar, Iyad Abdel Raouf Samir.
- **KArSL** — Sidig, Luqman, Mahmoud, Mohandes. *KArSL: Arabic Sign Language Database.*
  ACM TALLIP 20(1), 2021. Sign videos and labels.
- **Word-level ArSL ST-Transformer** — Yousef Elkilany, MIT License.
- **MediaPipe** (Apache 2.0) and **ONNX Runtime Web** (MIT).

## Licence

Code: MIT ([LICENSE](LICENSE)). Model files keep the licences of their sources: the sentence models are
CC BY-NC-SA 4.0 (from Isharah) and the word model is MIT (Yousef Elkilany).
