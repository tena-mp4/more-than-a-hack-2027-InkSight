# Improved approach 5 — high-precision boundary veto

## Description

The temporal structured model first proposes spells. Any boundary without local support from the
learned boundary detector is removed, merging the less-supported adjacent spell. The method tests
whether a precision-first veto is preferable to rewarding candidates during decoding.

## Research basis (2025 or newer)

- Kim et al., **ELASTIC**, 2026, motivates physical candidate validation before accepting temporal
  alignments.
- Borowiecki et al., **“Intention Driven Identification of In-Possession Match Phases in
  Association Football through Temporal Graph Learning”**, 2026, explicitly reports that high
  frame-level F1 can conceal sequence fragmentation and boundary error.

No pre-2025 paper is claimed as the scientific basis.

## Elements

- Structured initial path.
- Local boundary-support test.
- Emission-based choice of which adjacent run to merge.
- Iterative spell consolidation.
