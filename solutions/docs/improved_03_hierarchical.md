# Improved approach 3 — hierarchical dead/team decoding

## Description

Ball status is decoded first as a binary alive/dead sequence. Team possession is decoded only
inside the resulting live intervals. This removes dead ball as an intermediate shortcut between
teams and permits different transition costs for stoppages and turnovers.

## Research basis (2025 or newer)

- Kim et al., **“PathCRF: Ball-Free Soccer Event Detection via Possession Path Inference from
  Player Trajectories”**, KDD 2026, motivates football-consistent structured paths that prohibit
  logically invalid transitions.
- Peral et al., WACV 2026, jointly studies ball state and possessor identification as distinct but
  related tasks.

No pre-2025 paper is claimed as the scientific basis.

## Elements

- Binary alive/dead Viterbi decoder.
- Independent home/away decoder inside live intervals.
- DFL-selected dead-state bias and transition cost.
- Minimum-run constraint.
