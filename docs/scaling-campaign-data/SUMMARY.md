# Scaling campaign — running summary

## MPQC T2 wall-time (single rank, PaRSEC)

| molecule | cold (occ1) | warm (occ2) |
|---|---|---|
| C2H6 | 7.732 | 3.827 |
| C3H8 | 31.272 | 17.965 |
| C4H10 | 76.853 | 22.268 |
| C5H12 | 233.05 | 39.62 |
| C6H14 | OOM | - |

## Repro WARM T2 (t-dependent only) vs MPQC warm — the fair comparison

| molecule | np1 | np2 | np4 | np8 | np16 | MPQC warm | ratio(np1) |
|---|---|---|---|---|---|---|---|
| C2H6 | 2.96 | 4.53 | 4.32 | 3.57 | 3.17 | 3.827 | 0.77x |
| C3H8 | 22.80 | 31.10 | 27.84 | 21.57 | 18.24 | 17.965 | 1.27x |
| C4H10 | 37.67 | 40.36 | 34.49 | 25.00 | 20.22 | 22.268 | 1.69x |
| C5H12 | 115.26 | 114.72 | 88.27 | 63.14 | 50.43 | 39.62 | 2.91x |

## Repro COLD T2 multi-rank scaling

| molecule | np1 | np2 | np4 | np8 | np16 | speedup |
|---|---|---|---|---|---|---|
| C2H6 | 12.4 | 12.0 | 9.8 | 7.9 | 6.6 | 1.9x |
| C3H8 | 66.3 | 63.7 | 48.7 | 35.3 | 30.5 | 2.2x |
| C4H10 | 155.5 | 132.2 | 96.0 | 64.3 | 48.9 | 3.2x |
| C5H12 | 391.5 | 313.7 | 217.9 | 142.1 | 111.7 | 3.5x |

## Findings
- Warm (steady-state) repro ≈ MPQC warm only for small molecules: warm ratio(np1) grows
  C2H6 0.77x (repro faster) → C3H8 1.27x → C4H10 1.69x → C5H12 2.91x. (ratio = repro/MPQC.)
  Warm barely rank-scales (work too small).
- Two distinct 'cold' magnitudes — keep separate: (a) the shipped proto45/TPD=8 DF-half-transform
  config (comparison.csv: repro ~80s / MPQC 7.7s = ~10x np1; the repro self-scales ~5.4x over np1→16,
  narrowing the repro-vs-MPQC gap to ~2x at np16 — the 5.4x is the repro's own speedup, not the gap); (b) the §11
  whole-residual grid below (results.csv: repro cold np1 12.2s / MPQC 7.7s = ~1.6x). ~10x is (a), not (b).
- proto=100 generator extent avoids the giant intermediate (3x cold np1) but the t-dep value differs
  (MPQC_COMPARISON.md §11 reports ~0.2% cc-pVTZ order-sensitivity; the earlier ~7% is superseded).
- Same TiledArray fork cd53bd3 + same SeQuant derivation both sides; difference is evaluation/tiling.
- Hexane single-rank MPQC OOMs unbatched (63GB); with aux-Κ batching both MPQC and the repro
  complete hexane single-rank (repro T2 933s, peak 27.5GB — see MPQC_COMPARISON.md §11).
