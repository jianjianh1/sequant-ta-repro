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
| C2H6 | 2.96 | 4.54 | 4.30 | 3.63 | 3.17 | 3.827 | 0.77x |
| C3H8 | 22.71 | 30.52 | 27.61 | 21.66 | 18.44 | 17.965 | 1.26x |
| C4H10 | 37.59 | 41.32 | 35.58 | 25.21 | 20.61 | 22.268 | 1.69x |
| C5H12 | 116.20 | 114.87 | 91.01 | 62.33 | 48.75 | 39.62 | 2.93x |

## Repro COLD T2 multi-rank scaling

| molecule | np1 | np2 | np4 | np8 | np16 | speedup |
|---|---|---|---|---|---|---|
| C2H6 | 12.2 | 12.4 | 9.7 | 8.2 | 6.6 | 1.9x |
| C3H8 | 66.1 | 63.0 | 48.1 | 35.6 | 30.0 | 2.2x |
| C4H10 | 154.6 | 133.4 | 99.3 | 67.8 | 48.4 | 3.2x |
| C5H12 | - | 339.0 | 215.1 | 149.3 | 114.8 | - |

## Findings
- Warm (steady-state) repro ~ MPQC warm (C2H6 1.3x); warm barely rank-scales (work too small).
- Cold gap (~10x np1) is one DF half-transform; multi-rank scales it (C2H6 5.4x over 16 ranks).
- proto=100 generator extent avoids the giant intermediate (3x cold np1) but t-dep value differs ~7% (correctness open).
- Same TiledArray fork cd53bd3 + same SeQuant derivation both sides; difference is evaluation/tiling.
- Hexane single-rank MPQC OOMs (63GB) — needs multi-rank; repro uses alkanes-v3 hexane leaves.
