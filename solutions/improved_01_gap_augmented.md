# Improved approach 1 — target-shaped gap augmentation

## Description

The DFL emission model is retrained with contiguous missing-ball gaps sampled from the unlabelled
challenge tracking. This replaces independent random dropout with the observed target pattern: 150
gaps, median 268 frames and 90th percentile 845 frames. Clean and corrupted DFL sequences are both
used for training, followed by the DFL-selected global spell decoder.

## Research basis (2025 or newer)

- Peral et al., **“Multi-Modal Soccer Scene Analysis with Masked Pre-Training”**, WACV 2026. Its
  CropDrop mechanism motivates modality masking to prevent dependence on consistently visible ball
  information.
- Bassek et al., **“An integrated dataset of spatiotemporal and event data in elite soccer”**,
  *Scientific Data*, 2025, supplies the labelled DFL matches.

No pre-2025 paper is claimed as the scientific basis.

## Elements

- Unsupervised target missingness measurement.
- Contiguous domain-corruption augmentation.
- Two histogram gradient-boosting classifiers.
- Structured Viterbi spell decoder.
