# DFL approach 4 — phase-conditioned boundary model

## Description

This combines semantic Smart Tagging emissions with DFL boundary probabilities. Starts and ends of
set-piece and phase intervals supply broad, one-second boundary priors; attacking and defensive
transition intervals increase turnover likelihood. All cues remain soft and a global decoder
chooses one coherent sequence of possession spells.

## Scientific basis (2025 or newer)

- Peral et al., VISAPP 2025, for temporal context around player possession.
- Kim et al., **PathCRF**, KDD 2026, for constrained global paths.
- Kim et al., **ELASTIC**, 2026, for candidate-boundary evidence followed by global alignment.

No pre-2025 publication is used as the claimed basis.

## Elements

- DFL-supervised emission and boundary models.
- Labelled tactical phase regimes and set-piece types.
- Phase-aware team polarity.
- Soft boundary priors and structured decoding.
