# Five-method improvement benchmark

## Literature screening

Five implementable ideas survived screening: target-shaped masking, temporal context, hierarchical
state decoding, sparse candidate transitions and boundary veto. Three were rejected before
implementation:

- A full tactical-video transformer: the DFL training set has no video, so this could not be
  trained or validated without forbidden target labels.
- Direct PathCRF player-edge reproduction: the challenge ATD has anonymous fragmented slots rather
  than stable player identities. Its structured-decoding principle is retained instead.
- Reinforcement learning: there is no valid interactive environment or external reward; optimizing
  on the public labels would be test-set training.

Only work published in 2025 or later is used as the claimed scientific basis in the five approach
documents.

## Protocol

- Seven-match leave-one-match-out DFL validation.
- Contiguous DFL corruption sampled from the unlabelled challenge gap distribution.
- Model and decoder selection by masked-DFL spellF1.
- Public labels loaded only after all configurations and predictions were frozen.
- Random seeds fixed; every generated CSV passes the repository validator.

## Results

| Method | Masked DFL CV spellF1 | Public spellF1 | Public macroF1 | Predicted spells |
|---|---:|---:|---:|---:|
| Gap augmentation | 0.4056 | 0.3378 | 0.6950 | 28 |
| Temporal + semantic | 0.4282 | 0.3545 | 0.7178 | 27 |
| Hierarchical | 0.4152 | 0.4178 | 0.7260 | 28 |
| **Sparse boundary** | **0.4298** | **0.5100** | **0.7428** | **23** |
| Boundary veto | 0.3676 | 0.2065 | 0.4703 | 14 |

The best method raises the observed public spellF1 from 0.3909 to 0.5100. The 0.70 target is not
reached. Sparse candidates reduce false spell creation while retaining 13/17 true spells at IoU
0.3 and 12/17 at IoU 0.5. Remaining error is concentrated in strict boundary placement and the
four true spells missed even at loose IoU.

## Reproduction

```bash
python -m pip install -r requirements.txt
python solutions/run_improved_experiments.py
```

Machine-readable outputs are under `evaluation/results/improved_five/`.
