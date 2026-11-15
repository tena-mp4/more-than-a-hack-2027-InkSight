# Approach 2 — Temporally filtered evidence with hysteresis

## Solution description

This approach converts the same tracking features into two continuous signals:

1. team control, positive for home and negative for away;
2. dead-ball likelihood.

Team control is the bounded difference between the nearest-away and nearest-home player distances,
plus a small automatic team-tag prior. Both signals are filtered with exponential moving averages.
The decoder then uses hysteresis: the evidence required to enter a state differs from the evidence
required to remain in or leave it. Consequently, weak oscillations around zero do not repeatedly
split a possession.

Only visible-ball or tagged-team frames update the team filter. Missing observations retain the
previous filtered level. Dead-ball evidence is updated independently so team control and game
status are not conflated.

## Scientific basis (2025 or later only)

- Peral et al., **“Temporally Accurate Events Detection Through Ball Possessor Recognition in
  Soccer”**, VISAPP 2025. Their pipeline estimates per-frame possession likelihoods, combines them
  with temporal context, smooths noisy inputs, and rejects short false possession sequences. This
  implementation keeps that likelihood-plus-temporal-context principle but uses transparent
  filters instead of their supervised video network.
  <https://doi.org/10.5220/0013124700003912>
- Mills et al., **“Automatic event detection in association football using broadcast-derived
  tracking data”**, *Sports Engineering*, 2026. This supports using relative player–ball distance
  as possession evidence and treating set-piece/game-status logic separately.
  <https://doi.org/10.1007/s12283-026-00549-4>

No pre-2025 publication is used as the claimed scientific basis for this approach.

## Elements used

- Continuous distance-margin possession score.
- Hyperbolic tangent to bound extreme distance differences.
- Exponential temporal filters; `alpha=0.08` for team and `0.12` for dead/live.
- Hysteresis thresholds for switching teams and entering/exiting `D`.
- Missing-observation gating.
- Automatic Smart Tagging as a weak prior.
- No supervised ML, HMM, reinforcement learning, or model fitting.
- NumPy and Python standard library only.

## Why this is separate from ordinary smoothing

A centred majority window always spreads evidence symmetrically and can move both sides of a
boundary. This model maintains a causal belief and uses different thresholds for entry and exit.
Its weakness is that repeated moderate evidence can still accumulate into false switches.

## Reproduction

```bash
python solutions/run_approaches.py
```

Output:
`evaluation/results/top3/approach_2_hysteresis.csv`.

## Limitations

- The filter rates imply a fixed temporal scale.
- It is causal and cannot use strong future evidence to revise an earlier ambiguous boundary.
- It can fragment play if team evidence alternates for long enough.
- The thresholds are literature/domain defaults, not learned calibration probabilities.
