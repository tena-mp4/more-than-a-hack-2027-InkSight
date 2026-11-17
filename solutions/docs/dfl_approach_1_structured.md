# DFL approach 1 — supervised emissions with structured decoding

## Description

Seven public DFL matches provide exact `BallStatus` and `BallPossession` labels. Two histogram
gradient-boosting models estimate dead/alive and home/away control from provider-invariant tracking
geometry. Training includes 41% synthetic feature masking. A global Viterbi decoder converts these
emissions into possession spells; its switch cost was selected by leave-one-match-out DFL spellF1.

## Scientific basis (2025 or newer)

- Bassek et al., **“An integrated dataset of spatiotemporal and event data in elite soccer”**,
  *Scientific Data*, 2025. This supplies the official labelled training matches.
- Kim et al., **“PathCRF: Ball-Free Soccer Event Detection via Possession Path Inference from
  Player Trajectories”**, KDD 2026. This supplies the structured possession-path principle.

No pre-2025 publication is used as the claimed basis.

## Elements

- DFL XML streaming parser and cached NumPy features.
- Histogram gradient boosting, separated into dead/alive and team-control tasks.
- Missing-ball augmentation.
- CRF-like Viterbi maximum-score decoding with direct team-to-team transitions.
- Short-run constraint preventing one-frame dead-ball bridges.
