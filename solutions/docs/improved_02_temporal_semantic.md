# Improved approach 2 — temporal-context and semantic-phase fusion

## Description

Approach 1 is extended with past and future observations at 0.4 and 1.2 seconds, plus frames since
and until a visible ball observation. Phase-aware Smart Tagging supplies weak team and set-piece
priors. Defensive phase tags are inverted rather than incorrectly assigned to the defending team.

## Research basis (2025 or newer)

- Peral et al., **“Temporally Accurate Events Detection Through Ball Possessor Recognition in
  Soccer”**, VISAPP 2025, motivates contextual possessor evidence for precise event localisation.
- Peral et al., **“Multi-Modal Soccer Scene Analysis with Masked Pre-Training”**, WACV 2026,
  motivates temporal and cross-modal context under ball occlusion.

No pre-2025 paper is claimed as the scientific basis.

## Elements

- Sparse bidirectional temporal features.
- Explicit gap-age features.
- Semantic Smart Tagging fusion.
- DFL-supervised emissions and structured decoding.
