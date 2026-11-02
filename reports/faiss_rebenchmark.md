# T54 FAISS re-benchmark

Gold crops from T51's cache; 3 self-match crops excluded (T53). 28 gold classes are present in the index (`faiss_present_brands.csv`).

## All classes

Present 518 crops, absent 12205.

| k | accuracy (present) | n |
|---|---|---|
| 1 | 0.643 | 518 |
| 3 | 0.728 | 518 |
| 5 | 0.751 | 518 |

MRR (present): 0.685. On `absent` accuracy is 0 by construction (the true brand is not in the index); its top-1 similarity distribution is the abstain evidence:

| population | n | p5 | p25 | p50 | p75 | p95 |
|---|---|---|---|---|---|---|
| present, correct top-1 | 333 | 0.8969 | 0.9261 | 0.9403 | 0.9569 | 0.9717 |
| present, incorrect top-1 | 185 | 0.8903 | 0.9062 | 0.9253 | 0.9435 | 0.9682 |
| absent, top-1 | 12205 | 0.84 | 0.893 | 0.919 | 0.9405 | 0.9623 |

Top-1 similarity histogram, [0.8, 1.0]:

| bin | present correct | present incorrect | absent |
|---|---|---|---|
| [0.8, 0.81) | 0 | 0 | 112 |
| [0.81, 0.82) | 0 | 0 | 144 |
| [0.82, 0.83) | 0 | 0 | 147 |
| [0.83, 0.84) | 0 | 0 | 209 |
| [0.84, 0.85) | 1 | 1 | 273 |
| [0.85, 0.86) | 0 | 1 | 360 |
| [0.86, 0.87) | 3 | 0 | 353 |
| [0.87, 0.88) | 3 | 3 | 474 |
| [0.88, 0.89) | 5 | 4 | 719 |
| [0.89, 0.9) | 8 | 14 | 983 |
| [0.9, 0.91) | 20 | 33 | 1168 |
| [0.91, 0.92) | 27 | 25 | 1297 |
| [0.92, 0.93) | 28 | 25 | 1392 |
| [0.93, 0.94) | 69 | 25 | 1462 |
| [0.94, 0.95) | 30 | 29 | 1393 |
| [0.95, 0.96) | 89 | 5 | 922 |
| [0.96, 0.97) | 29 | 11 | 670 |
| [0.97, 0.98) | 18 | 6 | 115 |
| [0.98, 0.99) | 3 | 3 | 11 |
| [0.99, 1.0) | 0 | 0 | 1 |

Threshold sweep (present, top-1):

| threshold | predictions ≥ t | precision | recall |
|---|---|---|---|
| 0.8 | 518 | 0.643 | 0.643 |
| 0.82 | 518 | 0.643 | 0.643 |
| 0.84 | 518 | 0.643 | 0.643 |
| 0.86 | 515 | 0.645 | 0.641 |
| 0.88 | 506 | 0.644 | 0.629 |
| 0.9 | 475 | 0.659 | 0.604 |
| 0.92 | 370 | 0.719 | 0.514 |
| 0.94 | 223 | 0.758 | 0.326 |
| 0.96 | 70 | 0.714 | 0.097 |
| 0.98 | 6 | 0.500 | 0.006 |
| 1.0 | 0 | — | 0.000 |

**Overlap:** the distributions do not separate (no threshold reaches precision 0.95 on at least 10 predictions).

## Without channel classes (`evaluation.channel_classes`)

Present 518 crops, absent 6392.

| k | accuracy (present) | n |
|---|---|---|
| 1 | 0.643 | 518 |
| 3 | 0.728 | 518 |
| 5 | 0.751 | 518 |

MRR (present): 0.685. On `absent` accuracy is 0 by construction (the true brand is not in the index); its top-1 similarity distribution is the abstain evidence:

| population | n | p5 | p25 | p50 | p75 | p95 |
|---|---|---|---|---|---|---|
| present, correct top-1 | 333 | 0.8969 | 0.9261 | 0.9403 | 0.9569 | 0.9717 |
| present, incorrect top-1 | 185 | 0.8903 | 0.9062 | 0.9253 | 0.9435 | 0.9682 |
| absent, top-1 | 6392 | 0.8333 | 0.8825 | 0.9148 | 0.9375 | 0.9612 |

Top-1 similarity histogram, [0.8, 1.0]:

| bin | present correct | present incorrect | absent |
|---|---|---|---|
| [0.8, 0.81) | 0 | 0 | 79 |
| [0.81, 0.82) | 0 | 0 | 93 |
| [0.82, 0.83) | 0 | 0 | 103 |
| [0.83, 0.84) | 0 | 0 | 166 |
| [0.84, 0.85) | 1 | 1 | 223 |
| [0.85, 0.86) | 0 | 1 | 268 |
| [0.86, 0.87) | 3 | 0 | 247 |
| [0.87, 0.88) | 3 | 3 | 321 |
| [0.88, 0.89) | 5 | 4 | 430 |
| [0.89, 0.9) | 8 | 14 | 493 |
| [0.9, 0.91) | 20 | 33 | 510 |
| [0.91, 0.92) | 27 | 25 | 603 |
| [0.92, 0.93) | 28 | 25 | 688 |
| [0.93, 0.94) | 69 | 25 | 762 |
| [0.94, 0.95) | 30 | 29 | 670 |
| [0.95, 0.96) | 89 | 5 | 381 |
| [0.96, 0.97) | 29 | 11 | 252 |
| [0.97, 0.98) | 18 | 6 | 97 |
| [0.98, 0.99) | 3 | 3 | 6 |
| [0.99, 1.0) | 0 | 0 | 0 |

Threshold sweep (present, top-1):

| threshold | predictions ≥ t | precision | recall |
|---|---|---|---|
| 0.8 | 518 | 0.643 | 0.643 |
| 0.82 | 518 | 0.643 | 0.643 |
| 0.84 | 518 | 0.643 | 0.643 |
| 0.86 | 515 | 0.645 | 0.641 |
| 0.88 | 506 | 0.644 | 0.629 |
| 0.9 | 475 | 0.659 | 0.604 |
| 0.92 | 370 | 0.719 | 0.514 |
| 0.94 | 223 | 0.758 | 0.326 |
| 0.96 | 70 | 0.714 | 0.097 |
| 0.98 | 6 | 0.500 | 0.006 |
| 1.0 | 0 | — | 0.000 |

**Overlap:** the distributions do not separate (no threshold reaches precision 0.95 on at least 10 predictions).

## Recommendation

- Rule 7 (0.97 / gap 0.08), FAISS clause alone: fires on 17 present crops (precision 1.000, recall 0.033) and on no absent crop; precision over all firings 1.000 — supported by this evidence.
- `faiss.abstain_threshold`: the 95th percentile of absent top-1 similarity is 0.9612; abstaining below it keeps 0.085 of present crops correct at top-1. Set it only if that recall is acceptable.
- Kill criterion: correct and incorrect top-1 similarities do not separate, so FAISS should stay a hint (Rules 12/14 review it) and never decide.
- Rule 7 remains disabled unless the client agrees to a threshold from this evidence (T54).
