# Sparse boundary decoder: iteration 2

## Decision

Reject the peak-and-duration variant. The deployed model remains **sparse high-confidence
boundary decoder v1**, with public `spellF1 = 0.5100`.

The decision was made before inspecting the public five-minute result: on seven-match DFL
leave-one-match-out validation, v2 scored `0.43210`, below the paired v1 control at `0.43322`.
The frozen public diagnostic agreed with the rejection: v2 scored `0.4000` and v1 reproduced
`0.5100`. V2 predicted 27 spells versus 23 for v1 and 17 in the public ground truth.

## Reproducibility correction

Investigation found that v1's 0.510 result depended on an in-place array mutation. Running the
hierarchical decoder conditioned H/A probabilities on an inferred live-ball interval before the
sparse decoder executed. This was useful, but implicit and execution-order dependent.

`hierarchical_alive_calibrate` now expresses that cascade directly, and
`hierarchical_decode` copies its probability slice before normalisation. The explicit baseline
prediction has the same SHA-256 as the original winning CSV:

`cd9eb782879774b943a3a3014c89ad700071639864984b2b7bf346828ec1fa22`

## Tested change

V2 keeps the gradient-boosted emission model, boundary classifier, hierarchical live-ball prior,
and Viterbi decoder. It adds:

- moving-average boundary smoothing;
- temporal non-maximum suppression of correlated boundary peaks;
- a narrow low-cost switch window around each retained peak;
- minimum-duration merging as a lightweight semi-Markov approximation.

The grid was declared in `run_sparse_iteration.py`. It was selected only with target-shaped
missing-ball augmentation and leave-one-DFL-match-out `spellF1`; public labels were opened only
after predictions were frozen. The selected candidate used threshold `0.55`, smoothing radius
`2`, switch radius `6`, peak separation `20`, minimum team duration `10`, and minimum dead-ball
duration `8` frames.

## Scientific grounding

- [PathCRF: Event Boundary-Aware Temporal Graph Learning for Soccer Possession
  Detection (2026)](https://arxiv.org/abs/2602.12080) motivates explicit boundary-aware structured
  decoding rather than independent frame classification.
- [Multi-Modal Soccer Scene Analysis with Masked Pre-Training
  (WACV 2026)](https://openaccess.thecvf.com/content/WACV2026/papers/Peral_Multi-Modal_Soccer_Scene_Analysis_with_Masked_Pre-Training_WACV_2026_paper.pdf)
  supports temporal modelling under missing modalities, reflected here by contiguous
  target-shaped masking and bidirectional context.

The negative result is informative: generic peak suppression and short-duration constraints do
not transfer across the DFL/challenge provider shift. They slightly increased fragmentation on
the target. Further work should improve boundary-domain adaptation or exploit an independently
labelled target-provider match, not tune these parameters against the public five minutes.

## Run

```bash
python solutions/run_sparse_iteration.py
```

Outputs, including the comparison plot, CV rows, selected configuration, predictions, and frozen
public metrics, are in `evaluation/results/sparse_iteration_v2/`.
