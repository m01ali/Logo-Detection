# T51 FAISS self-match diagnostic

12726 gold crops queried (46 skipped with a side below 16 px).

- Self-match (raw top-1 similarity ≥ 0.999): **3**
- Near-duplicate band (0.99–0.999): 1
- Below 0.99: 12722

**Caveat.** Adjacent-frame crops of the same bug (a static channel logo) score very high without being the same crop, so the near-duplicate band is not evidence of leakage; only ≥ 0.999 is treated as a self-match.

## Raw top-1 similarity

| bin | count |
|---|---|
| [0.0, 0.05) | 0 |
| [0.05, 0.1) | 0 |
| [0.1, 0.15) | 0 |
| [0.15, 0.2) | 0 |
| [0.2, 0.25) | 0 |
| [0.25, 0.3) | 0 |
| [0.3, 0.35) | 0 |
| [0.35, 0.4) | 0 |
| [0.4, 0.45) | 0 |
| [0.45, 0.5) | 0 |
| [0.5, 0.55) | 0 |
| [0.55, 0.6) | 0 |
| [0.6, 0.65) | 0 |
| [0.65, 0.7) | 0 |
| [0.7, 0.75) | 0 |
| [0.75, 0.8) | 31 |
| [0.8, 0.85) | 856 |
| [0.85, 0.9) | 2930 |
| [0.9, 0.95) | 7023 |
| [0.95, 1.0) | 1886 |

## Matched brand of the self-match cases

| gold class | matched index brand | n |
|---|---|---|
| como calcio | Etro | 3 |

## `verisure`

22 crops; self-match 0, near-duplicate 0; median raw top-1 0.9445.

## Conclusion (T52)

3 crops self-match (listed in `faiss_selfmatch_excluded.csv`); recommend excluding them from the benchmark; a rebuild is not proposed.
