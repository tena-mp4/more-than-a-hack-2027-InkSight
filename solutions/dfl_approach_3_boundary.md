# DFL approach 3 — learned transition-boundary fusion

## Description

A second DFL-supervised gradient-boosting model predicts whether each frame lies within two frames
of a team, stoppage, or restart transition. Its probability reduces the global decoder's switching
penalty locally. This tests whether explicitly detecting boundaries improves possession-spell
localisation over emissions and duration regularisation alone.

## Scientific basis (2025 or newer)

- Kim et al., **“ELASTIC: Trajectory-Based Synchronization of Event and Tracking Data in Soccer”**,
  2026. ELASTIC first identifies physically plausible temporal candidates and then performs global
  sequence alignment.
- Peral et al., VISAPP 2025, for temporally precise possession-derived action boundaries.

No pre-2025 publication is used as the claimed basis.

## Elements

- Imbalance-aware binary gradient boosting.
- Past/future distance-margin, ball-motion, reach-time, and proximity features.
- Boundary-conditioned transition costs.
- Global Viterbi decoding and minimum-run constraints.
