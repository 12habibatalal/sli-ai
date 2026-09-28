# Sentence model rounds

Word error rate (WER, lower is better) on Isharah-1000, signer-independent split (SI): 10 train
signers, 1 dev signer (model selection), 4 test signers (evaluated once, at the end).

## Targets

- **Stopping target (spec):** the Isharah paper reports no pose-based baseline, so its best SI
  baseline of any kind: Swin-MSTP (RGB video), **dev 17.9 / test 26.6**. Source: Alyami et al.,
  arXiv 2506.03615, Table 4 (US: best SMKD dev 56.6 / test 48.0).
- **For reference, pose-based results on the same splits (MSLR 2025 challenge,** arXiv 2508.09372**):**
  CNN-BiLSTM baseline dev 14.54 / test 22.62; best entry (Signer-Invariant Conformer) dev 7.31 /
  test 13.07. US: best dev 55.08 / test 47.78.

## Rounds

| Round | Date | Model | Change (one thing) | Epochs | Dev WER | Kaggle GPU h | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2026-09-28 | large | KArSL isolated-sign pre-training (502 signs) | 30 | 0.032 (KArSL test, seen signers) | 0.7 (baraasaad) | init for Isharah rounds |
| 1 | 2026-09-28 | large | Isharah SI, init from round 0 | 60 | **0.262** | 1.8 (baraasaad) | train loss 48 -> 0.016 while dev flat from epoch ~40: overfits the 10 training signers |
| 2 | 2026-09-28 | large | no pre-training (vs round 1) | 60 | 0.893 | 1.6 (mono768) | never leaves the all-blank solution (loss stuck ~5.2): pre-training is required |
| 3 | 2026-09-28 | large | strong augmentation (vs round 1) | 60 | running | | rot 25/15, scale 0.3/0.2, speed 0.6-1.4, 2x longer/more hand drops |
| 4 | 2026-09-28 | large | dropout 0.1 -> 0.3 (vs round 1) | 60 | running | | parallel with 3 on mono768 |
