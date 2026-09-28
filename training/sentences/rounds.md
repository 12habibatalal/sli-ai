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
