# Approach 3 — Constrained global Viterbi decoder

## Solution description

This method treats an entire half as one structured sequence. Each frame receives three emission
scores for `H`, `A`, and `D` from nearest-player distance, ball location, set-piece phase, and
automatic team tags. Viterbi dynamic programming selects the single highest-scoring label path.

Transitions carry explicit costs:

- `H ↔ A` costs more because it requires controlled possession by the opponent;
- transitions to or from `D` cost less because stoppages legitimately delimit possessions;
- remaining in the current state has no cost.

Where the ball is missing, emissions are mostly neutral, so the global path continues the existing
state rather than manufacturing a label from missingness. Unlike causal hysteresis, later evidence
can determine the best state through an earlier ambiguous interval.

This is a deliberately small linear-chain approximation, not a reproduction of the much larger
neural PathCRF model.

## Scientific basis (2025 or later only)

- Kim et al., **“PathCRF: Ball-Free Soccer Event Detection via Possession Path Inference from
  Player Trajectories”**, 2026 preprint. PathCRF formulates soccer possession as structured path
  inference, constrains invalid transitions, and uses Viterbi decoding to obtain a logically
  consistent possession sequence. This implementation adopts those three architectural ideas at
  team-state level without training a graph neural network.
  <https://arxiv.org/abs/2602.12080>
- Mills et al., **“Automatic event detection in association football using broadcast-derived
  tracking data”**, *Sports Engineering*, 2026. This supports the tracking-derived proximity and
  set-piece emissions.
  <https://doi.org/10.1007/s12283-026-00549-4>

No pre-2025 publication is used as the claimed scientific basis for this approach.

## Elements used

- Three-state linear-chain structured model (`H`, `A`, `D`).
- Hand-specified emission scores; no label fitting.
- Transition-score matrix encoding football constraints.
- Global Viterbi dynamic programming, independently for each half.
- A CRF-like maximum-score decoder, but no trained CRF parameters.
- Missing observations represented by low-information emissions.
- NumPy and Python standard library only.
- No reinforcement learning or neural framework.

## Relation to an HMM and CRF

The implementation has HMM-like hidden states and CRF-like feature scores. It does not estimate
normalised generative probabilities, so calling it a trained HMM would be inaccurate. It is best
described as a fixed-score linear-chain model decoded by Viterbi.

## Reproduction

```bash
python solutions/run_approaches.py
```

Output:
`evaluation/results/top3/approach_3_viterbi.csv`.

## Limitations

- Fixed transition costs are not duration-aware; a semi-Markov decoder could model spell lengths
  more explicitly.
- Emissions remain dependent on noisy ball tracking when it is present.
- Global decoding is acausal and therefore intended for offline submission generation.
- This compact approximation should not be presented as implementing the full PathCRF paper.
