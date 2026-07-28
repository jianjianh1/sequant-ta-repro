# §11 scaling-campaign data & tooling

Raw evidence and scripts behind `docs/MPQC_COMPARISON.md` §11 (alkane series
C₂H₆–C₅H₁₂, 1-rank-per-node sweep np∈{1,4,8,16} over CloudLab node16–31, both
sides). Numbers here are the source of the §11 tables.

## Data (CSV)

- `comparison.csv` — MPQC single-rank (np=1) cold/warm T2 by molecule + a few repro summary rows.
- `results_warm.csv` — repro **warm** (t-dependent only, `ta_warm_t2`) raw per-trial T2, all molecules × np.
- `results.csv` — repro **cold** (whole residual) raw per-trial T1/T2, all molecules × np.
- `mpqc_mr.csv` — MPQC **multi-rank** (np{4,8,16}) cold(occ1)/warm(occ2) T2 — the fair same-np comparison.
- `correctness_anchors.csv` — gauge-free R(T=0) repro-vs-MPQC-occ1 checksums (C3H8/C4H10 bit-exact; C5H12 ~7 sig figs).
- `SUMMARY.md` — regenerated tables (via `scripts/make_summary.py`).

## Scripts (CloudLab-specific; paths/hostnames hardcoded to this experiment)

- `make_mpqc_variants.py` — derive per-molecule MPQC ref/perf input JSONs from the alkanes-v3 templates.
- `repro_sweep.sh` / `warm_sweep.sh` — repro cold / warm np-sweeps (stage leaves to `/local`, `mpirun --bind-to none`, descending np).
- `mpqc_mr_sweep.sh` — MPQC multi-rank np-sweep via `run-mpqc-mpirun.sh` (PaRSEC; sif on shared `/proj`).
- `make_summary.py` / `make_fair.py` — regenerate the SUMMARY / fair same-np ratio tables from the CSVs.

## Key config (see §11 + memory `scaling-campaign-harness`)

- Repro binaries: built `-DSPTC_OWNING_TOT` (owning-ToT — required at multi-rank; ArenaTensor
  segfaults at np≥8 for big molecules). Run config `SPTC_COARSE_OCC=9 SPTC_OCC_TILE=2
  SPTC_COARSE_PAD=0 SPTC_TILES_PER_DIM=8 SPTC_MAD_WAIT_POLICY=yield MAD_NUM_THREADS=8`,
  `mpirun --bind-to none --mca btl_tcp_if_include 10.10.1.0/24`.
- Leaves: `jianjianh1/mpqc-alkanes-v3` (cc-pVTZ), converted via `mpqc-benchmark/bin/tns-to-sptc-coo.py`.
- Both sides use TiledArray fork commit `cd53bd3` + the same SeQuant derivation.

## Headline

At equal ranks MPQC is faster for every real molecule (repro ~2–3× slower warm, ~4.5–7×
cold at np=16); the repro leads only on tiny ethane's warm residual. The gap is the giant
DF-half-transform intermediate (repro materialises it → scales worse + hexane memory wall;
MPQC distributes it). See §11 for the full analysis and the proto=100 open lever.
