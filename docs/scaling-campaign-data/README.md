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

Additional data committed since this README's first draft:
- **Deep profile (2026-08-02/03, `docs/MPQC_PROFILE_DEEP.md`)** — `hwcounters.csv` (IPC + DRAM bandwidth),
  `thread_sweep.csv` (per-thread-count self-time), `warm_profile.csv`, `comm_profile.csv` (real-np comm/compute),
  `mpqc_counters.csv` (MPQC-side IPC/BW), + `profiles/` (flame graphs, per-rank self-time).
- **Hotspot (`docs/GAP_RESEARCH.md`)** — `gap_ceiling.csv` (hand-GEMM vs einsum ceiling), `gap_decomposition.csv`
  (single-rank × scaling gap split), `gap_profile.txt` (perf self-time + einsum buckets).
- **Refuted-lever diagnostics** — `occ_tiling_experiment.csv`, `cyclic_pmap_timing.csv`, `parsec_experiment.csv`,
  `pmap_distribution_C4H10_np16.txt`.
- **Aux-Κ batching / hexane** — `auxbatch_correctness.csv`, `repro_hexane.csv`, `repro_hexane_batch.csv`, `mpqc_hexane_batch.csv`.
- **Contraction IR** — `whole_t1_residual.ctir`, `whole_t2_residual.ctir` (→ `docs/CONTRACTION_IR.md`).

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

At equal ranks MPQC is faster for every real molecule (repro ~2–3× slower warm, **~4.5–5.5×
cold at np=16**, rising to ~7× at np=4); the repro leads only on tiny ethane's warm residual.
The gap is the giant DF-half-transform intermediate's SUMMA — both sides materialise it; MPQC
distributes/lays it out better across ranks (and it is a hexane memory wall for both). See §11
for the full analysis; the counter/comm-measured verdict is in `docs/MPQC_PROFILE_DEEP.md`
(the "~87 %/~100×-off-peak" mechanism it once cited is superseded there).
