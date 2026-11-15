# Improved approach 4 — sparse high-confidence boundary decoding

## Description

A DFL-supervised boundary classifier proposes a sparse set of high-confidence transition frames.
Switching is expensive elsewhere and cheaper only at candidates. This directly targets false spell
creation instead of smoothing every emission equally.

## Research basis (2025 or newer)

- Kim et al., **“ELASTIC: Trajectory-Based Synchronization of Event and Tracking Data in Soccer”**,
  2026, motivates extracting sparse physically plausible candidate frames before global alignment.
- Kim et al., **PathCRF**, KDD 2026, motivates globally consistent decoding after candidate scoring.

No pre-2025 paper is claimed as the scientific basis.

## Elements

- Imbalance-aware gradient-boosted boundary classifier.
- Threshold 0.80 selected by masked leave-one-match-out DFL spellF1.
- Transition cost 14 outside candidates and 5 at candidates.
- Semantic phase emissions plus global structured decoding.
