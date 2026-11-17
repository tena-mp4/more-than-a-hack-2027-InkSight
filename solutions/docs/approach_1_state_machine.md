# Approach 1 — Controlled-possession state machine

## Solution description

This is the simplest of the three systems. It reads the tactical automatic tracking, converts
normalised coordinates to metres, and finds the two nearest players from each team whenever the
ball is observed. A team becomes a possession candidate when its nearest player is sufficiently
close to the ball and the home-versus-away distance evidence has a clear sign.

Possession does not switch on one frame. The candidate must persist for eight frames before the
state changes. This encodes the challenge rule that a touch, deflection, or isolated nearest-player
observation is not controlled possession. The current team is retained through uncertain and
missing-ball frames.

Dead-ball evidence comes from an observed ball outside the pitch, the automatic set-piece tags,
and the published half/kickoff boundaries. Dead-ball entry and exit also require confirmation.
The system never treats ball missingness alone as a dead ball.

## Scientific basis (2025 or later only)

- Mills et al., **“Automatic event detection in association football using broadcast-derived
  tracking data”**, *Sports Engineering*, 2026. The paper evaluates deterministic auto-eventing
  from player/ball tracking. Its relevant ideas are possession zones based on relative
  player–ball distance and football-logic triggers/patterns for set pieces.
  <https://doi.org/10.1007/s12283-026-00549-4>
- Peral et al., **“Temporally Accurate Events Detection Through Ball Possessor Recognition in
  Soccer”**, VISAPP 2025. The paper makes ball possessor recognition an intermediate temporal
  task and rejects short, spurious possession sequences before detecting events.
  <https://doi.org/10.5220/0013124700003912>

No pre-2025 publication is used as the claimed scientific basis for this approach.

## Elements used

- Deterministic finite-state machine with states `H`, `A`, and `D`.
- Metric player–ball distance and distance-margin evidence.
- Possession zone with a fixed 3 m radius.
- Eight-frame controlled-possession confirmation.
- Six-frame dead/live confirmation.
- Automatic Smart Tagging as a weak prior.
- No HMM, neural network, reinforcement learning, or fitted parameters.
- NumPy and Python standard library only.

## Missing data

Missing ball observations produce no new proximity evidence. They do not force `D`; the current
state survives until observed evidence supports a transition. This is important because ball
missingness is common and is not equivalent to a stoppage.

## Reproduction

Run all methods and the comparison:

```bash
python solutions/run_approaches.py
```

The resulting submission is
`evaluation/results/top3/approach_1_state_machine.csv`. All constants are written to
`evaluation/results/top3/frozen_config.json` before public scoring.

## Limitations

- A fixed possession radius cannot distinguish every interception from a deflection.
- Set-piece intervals are approximate and therefore only weak evidence.
- The counter introduces a small transition delay.
- It does not infer ball trajectories during long tracking gaps.
