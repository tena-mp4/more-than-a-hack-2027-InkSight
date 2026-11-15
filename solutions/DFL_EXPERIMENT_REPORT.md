# DFL experiment report

## Protocol

- Seven DFL matches; leave-one-match-out validation.
- Primary selection metric: repository `spellF1`.
- A second validation pass masks 41% of ball/control features.
- Decoder grid: 2.2, 3, 4, 5, 6, 7, 8; an edge check tested 10, 12, 14 and 16.
- The public five minutes were loaded only after predictions and configuration were frozen.

## Main results

The selected decoder cost is 8.0: clean DFL spellF1 0.6066 and masked DFL spellF1 0.5733.
Higher costs reduce both spellF1 and the predicted/true spell ratio.

On the frozen public diagnostic, the semantic-tag model is best of the new approaches at 0.3909
spellF1 and 0.7214 macroF1. The improvement over the old Viterbi spellF1 of 0.3882 is below the
documented 0.04 spell-level noise band and is not claimed as a significant win.

See `evaluation/results/dfl_four/public_detailed.json`, `decoder_cost_cv.json`,
`decoder_cost_masked_cv.json`, and `comparison.svg` for machine-readable results.

## Reproduction

```bash
python -m pip install -r requirements.txt
python solutions/run_dfl_experiments.py
```

The first invocation streams the seven XML files into versioned local caches. Subsequent runs reuse
the cache. `--skip-cv` reruns final fitting, inference and the frozen public diagnostic while
preserving the saved cross-validation results.
