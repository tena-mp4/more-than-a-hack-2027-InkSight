# DFL approach 2 — structured model with semantic Smart Tagging

## Description

Approach 1 is augmented with all 638 automatic phase intervals. Offensive phase tags support the
tagged team; defensive and defensive-transition tags support its opponent. Correlated overlapping
tags are aggregated into a signed confidence. Set pieces provide weak dead-ball evidence rather
than being copied as dead intervals. The fused probabilities are globally decoded into spells.

## Scientific basis (2025 or newer)

- Peral et al., **“Temporally Accurate Events Detection Through Ball Possessor Recognition in
  Soccer”**, VISAPP 2025. The relevant idea is combining temporal possessor evidence with event
  context rather than treating an interval label as exact possession.
- Kim et al., **PathCRF**, KDD 2026, for logically consistent global possession decoding.

No pre-2025 publication is used as the claimed basis.

## Elements

- Metrica automatic phase labels loaded by `starter/load_smart_tagging.py`.
- Phase-aware rule mapping, including inversion of defensive-team tags.
- Log-probability fusion with explicit conflict confidence.
- Structured spell decoder from approach 1.
