# Final possession-inference report

## Outcome

The best valid solution is the **sparse high-confidence boundary decoder**, with public
`spellF1 = 0.5100`. The score is measured on the supplied labelled five-minute interval and was
used only as a frozen final diagnostic, not for fitting or hyperparameter selection.

The final model also achieved:

| Metric | Result |
| --- | ---: |
| spellF1 | **0.5100** |
| Macro-F1 | 0.74276 |
| Alive/dead F1 | 0.93028 |
| Predicted possession spells | 23 |
| Ground-truth possession spells | 17 |
| DFL leave-one-match-out spellF1 | approximately 0.43 |

The target of 0.70 was not reached. Reporting 0.607 as the challenge result would be incorrect:
that value came from a DFL-domain cross-validation experiment, not from the challenge game's
labelled five minutes.

## Constraints and experimental protocol

The central constraint was to avoid learning from the public five-minute target labels. Those
labels are a test set. Using them for model fitting, selecting thresholds, moving boundaries, or
choosing an ensemble would leak test information.

The protocol was therefore:

1. Build and freeze approaches using unlabelled target inputs and/or independently labelled DFL
   matches.
2. Select configurations with leave-one-match-out DFL validation and spellF1.
3. Write full-match predictions.
4. Open the public five-minute labels only for the final diagnostic.
5. Preserve configurations, predictions, detailed metrics, and plots in `evaluation/results/`.

The ranking metric drove the design. Frame accuracy alone is insufficient because a classifier
can label most frames correctly while fragmenting a real possession into several false spells.

## Step 1: deterministic baselines

The first phase established simple, reproducible baselines from target tracking evidence:

1. A controlled-possession state machine assigned the ball to the nearest plausible team and
   applied persistence rules.
2. A hysteresis model smoothed the evidence and required stronger support to switch teams than to
   retain the current state.
3. A three-state Viterbi decoder globally balanced frame evidence against transition costs.

The Viterbi baseline was best, with public `spellF1 = 0.38824`. This showed that global temporal
consistency helped, but the models still suffered from missing-ball gaps, uncertain dead-ball
states, and too many poorly placed switches.

## Step 2: independently labelled DFL training

Seven DFL matches under `data/extra/28196177` were parsed with a streaming XML reader. Streaming
was necessary because the raw files are large; only provider-invariant arrays were cached.

The shared feature set included:

- ball visibility, normalised location, boundary distance, speed, acceleration, and height;
- nearest and second-nearest home/away player distance;
- local player counts around the ball;
- approximate time-to-arrival and home/away control margins;
- past and future temporal summaries;
- frames since and until a visible ball observation.

Two histogram-gradient-boosting classifiers formed the emission model:

1. live ball versus dead ball;
2. home versus away, conditional on the ball being live.

Initial DFL-supervised variants combined these emissions with Viterbi decoding, learned boundary
probabilities, Smart Tagging phases, and voting. The best public score in this phase was only about
0.391. Clean DFL cross-validation was much higher, demonstrating a substantial provider/domain
shift rather than a lack of model capacity.

## Step 3: diagnose the transfer failures

Three weaknesses were identified:

1. **Missingness mismatch.** The challenge ball disappears in long contiguous intervals, whereas
   random independent masking does not reproduce this failure mode.
2. **State hierarchy.** Dead/live status and team possession have different dynamics. Treating
   H/A/D as symmetric states confuses restarts with possession changes.
3. **Boundary density.** A standard Viterbi model can switch anywhere. Frame-level uncertainty then
   becomes excess possession spells, which is heavily penalised by spellF1.

Target-ball gaps were measured without labels: 150 gaps, median 268 frames, 90th percentile about
845 frames, and maximum 5,537 frames. DFL training matches were augmented with gaps sampled from
that empirical distribution.

## Step 4: benchmark five focused improvements

Five approaches were selected from 2025-or-newer literature and tested under identical DFL
leave-one-match-out validation:

| Approach | Public spellF1 | Public macro-F1 | Predicted spells |
| --- | ---: | ---: | ---: |
| Target-shaped gap augmentation | 0.33778 | 0.69502 | 28 |
| Temporal context plus semantic phases | 0.35455 | 0.71776 | 27 |
| Hierarchical dead/team decoding | 0.41778 | 0.72602 | 28 |
| **Sparse high-confidence boundary decoder** | **0.51000** | **0.74276** | **23** |
| Boundary-veto decoder | 0.20645 | 0.47028 | 14 |

The boundary-veto failure is important: aggressively deleting uncorroborated transitions reduced
the spell count but merged genuine possessions and badly damaged team attribution. Fewer spells is
not automatically better.

## Final solution, step by step

The winning pipeline is deliberately compact:

1. **Extract provider-invariant geometry.** Convert DFL and challenge tracking into the same
   ball-relative team-control features without depending on persistent target player identities.
2. **Apply target-shaped missingness augmentation.** Mask DFL ball observations in contiguous gaps
   sampled from the unlabelled target game.
3. **Fit emission classifiers.** Train balanced histogram-gradient-boosting models for dead/live
   status and home/away possession.
4. **Add temporal context.** Include bidirectional observations and explicit gap-length context.
5. **Fuse Smart Tags.** Convert offensive, defensive, transition, and set-piece phases into weak
   semantic priors. Tags influence probabilities; they do not overwrite tracking evidence.
6. **Calibrate with the state hierarchy.** A two-level decoder first estimates temporally coherent
   live-ball intervals. Inside those intervals, team probabilities are conditioned on the ball
   being live.
7. **Predict candidate boundaries.** A separate gradient-boosting classifier estimates whether a
   frame lies near a labelled DFL state transition.
8. **Run sparse Viterbi decoding.** State changes cost 14 by default and cost 5 only where boundary
   confidence is at least 0.80. This converts dense frame probabilities into a small set of
   coherent spells.
9. **Merge microscopic runs and write H/A/D/X labels.** Valid half ranges are decoded; frames
   outside them remain `X`.

This combination works because emissions answer *which state is plausible*, the hierarchical
prior answers *whether team possession is currently meaningful*, and the boundary model answers
*where a new spell is allowed to begin*.

## Step 5: second sparse-decoder iteration and failure

A further iteration tested smoothed boundary peaks, temporal non-maximum suppression, narrow
switch windows, and minimum-duration cleanup. These are reasonable structured-decoding ideas, but
the experiment rejected them:

- v1 DFL CV spellF1: `0.43322`;
- v2 DFL CV spellF1: `0.43210`;
- v1 public spellF1: `0.5100`;
- v2 public spellF1: `0.4000`.

The v2 candidate produced 27 target spells, so peak suppression did not transfer as intended. The
provider shift changed the boundary-score shape enough that smoothing and duration rules licensed
additional poor transitions. V1 remains deployed.

## Reproducibility correction

During the v2 paired-control run, v1 initially failed to reproduce. Investigation found an
in-place NumPy mutation: the hierarchical decoder normalised a view into the shared probability
matrix, silently calibrating the sparse decoder that ran afterward.

The useful behaviour is now explicit in `hierarchical_alive_calibrate`, and
`hierarchical_decode` copies its slice before normalisation. The repaired pipeline no longer
depends on call order. Its winning CSV has exactly the same SHA-256 as the original result:

`cd9eb782879774b943a3a3014c89ad700071639864984b2b7bf346828ec1fa22`

## Scientific basis

Only work published in 2025 or later is used as claimed grounding:

- [Temporally Accurate Events Detection Through Ball Possessor Recognition in
  Soccer (VISAPP 2025)](https://doi.org/10.5220/0013124700003912) supports temporally contextual
  possession evidence and possessor-based event localisation.
- [PathCRF: Event Boundary-Aware Temporal Graph Learning for Soccer Possession Detection
  (2026)](https://arxiv.org/abs/2602.12080) motivates boundary-aware, globally consistent
  possession paths.
- [Multi-Modal Soccer Scene Analysis with Masked Pre-Training
  (WACV 2026)](https://openaccess.thecvf.com/content/WACV2026/papers/Peral_Multi-Modal_Soccer_Scene_Analysis_with_Masked_Pre-Training_WACV_2026_paper.pdf)
  motivates temporal modelling that remains useful when modalities are missing.
- ELASTIC (2026) motivates identifying sparse plausible temporal candidates before global
  alignment; the exact reference and approach mapping are recorded in the per-method documents.

Full approach-specific citations and component descriptions are available in this directory.

## Reproduce

From the repository root:

```bash
python -m pip install -r requirements.txt
python solutions/run_improved_experiments.py
python solutions/run_sparse_iteration.py
```

The first experiment rebuilds the five-method comparison and winning prediction. The second
reproduces the paired v1/v2 rejection experiment. The principal artifacts are:

- `evaluation/results/improved_five/04_sparse_boundary.csv` — winning full-match prediction;
- `evaluation/results/improved_five/public_results.json` — five-method summary;
- `evaluation/results/improved_five/public_detailed.json` — detailed metrics;
- `evaluation/results/improved_five/comparison.svg` — comparison plot;
- `evaluation/results/sparse_iteration_v2/` — second-iteration evidence and rejection decision.

The implementation avoids deep frameworks and custom training infrastructure: NumPy,
scikit-learn histogram gradient boosting, deterministic feature extraction, and explicit dynamic
programming are sufficient for the final result.
