# Batched scale-GEMM patch — reproduces MPQC's per-pair GEMM for the μ̃Κ half-transform

> **EXPERIMENT — NOT part of the benchmark harness.** This is a TiledArray-backend kernel
> modification, env-gated by `SPTC_SCALE_GEMM=1` and **default-off** (unset = byte-identical stock
> backend). It is preserved here for reproducibility only. Do **not** enable it for a
> framework-comparison run — a hand-optimized backend is not a representative yardstick. See
> `docs/HARNESS_VS_EXPERIMENTS.md`.

Landed 2026-08-01. Env-gated by `SPTC_SCALE_GEMM=1` (default off = byte-identical baseline).

## Result (C2H6 cc-pVTZ, cold T2, 1 thread, node3, clang/OpenBLAS arena)
- baseline (gate off): **30.60 s**
- SPTC_SCALE_GEMM=1:   **20.01 s**  → **1.53×**, checksum exact (nnz=150301, sumsq/max_abs identical;
  sum differs at the ~13th digit = GEMM vs sequential-AXPY summation order, negligible).
- vs MPQC 8.26 s: gap 3.7× → **2.4×**. The 358.8 M per-cell scalar AXPY calls → **0** (all batched).

## What it does
The μ̃Κ half-transform `I(i,i,μ̃',Κ;a) = Σ_μ̃ g(μ̃,μ̃',Κ)·C(i,i,μ̃;a)` (generated_t2:487) and ~50
cousins have the PNO index `a` as a ToT inner *spectator* (broadcast) while contracting an *outer*
index (μ̃/Κ). TA dispatched these as per-(cell×k) scalar AXPY (`fused_scale_t_x_tot_inplace`) — 358 M
calls, ~1/3 of 1-thread wall. This patch adds a strided op that, per occupied pair, gathers R's
μ̃-run of inner-a vectors into `Rmat[K×Q]` and issues ONE GEMM `tmp[M×Q]=L[M×K]·Rmat`, scattering into
result cells — the same per-pair GEMM MPQC uses. Non-uniform pairs fall back to the correct per-cell
AXPY. Correctness gated by `ta_compute_checksum`; result cells are reserved by the arena plan before
the strided op runs.

## Files (apply over third_party/tiledarray-cd53bd3-clang/install/include/TiledArray/ and src/)
- `arena_einsum.h` — adds `scale_gemm_enabled()` (SPTC_SCALE_GEMM gate) + `arena_strided_scale`
  kernel (after `fused_scale_t_x_tot_inplace`); also carries the `[scale-fused]` TA_GEMM_TIMING
  counters used to attribute the cost.  → tensor/arena_einsum.h
- `cont_engine.h` — new member `arena_strided_scale_tile_op_`; setup in the `t_x_tot` scale branch of
  `init_inner_tile_op_owning_` (gated); install at all 3 `set_strided_oprod_op` sites (init_struct ×2,
  init_struct_general).  → expressions/cont_engine.h
- `ta_tensor_loader.h` — the (separate, refuted) SPTC_COMPACT_COEFFS compaction port + the compaction
  helper. Harmless (gated); kept for the record.  → repo src/ta_tensor_loader.h

Apply the patch to a disposable experimental TiledArray checkout, not to the
canonical `third_party/tiledarray-cd53bd3-clang` install.

## Reproduce
```
mpirun -np 1 --bind-to none \
  -x SPTC_COARSE_OCC=9 -x SPTC_OCC_TILE=2 -x SPTC_COARSE_PAD=0 -x SPTC_TILES_PER_DIM=8 \
  -x SPTC_MAD_WAIT_POLICY=yield -x MAD_NUM_THREADS=1 -x SPTC_TRIALS=1 -x SPTC_WARMUP=0 \
  -x SPTC_SCALE_GEMM=1 -x OPENBLAS_NUM_THREADS=1 -x OMP_NUM_THREADS=1 \
  <experiment-build>/ta_sequant_native_residual_main <leaf_dir>
```
Add `-x TA_GEMM_TIMING=1` to see the `[scale-fused]` counter drop to 0 when the path fires.

## Scope / next
Removes the fused_scale class (~10 s). Remaining ~20 s vs MPQC 8.26 s is other per-cell arena paths
(dgemm, the 26 ce+ce "non-canonical inner perm" reverts, arena grow/plan, other terms) — same shape,
would need the same batching treatment. Multi-rank: arena has a known np≥8 lazy-deletion segfault, so
this single-node result does not directly transfer to the distributed runs (owning ToT is shipped
there); the kernel is single-node-validated only.
