# SPTC_CE_E_GEMM — op-488 coalescing kernel (the work-coalescing lever, single-node)

> **EXPERIMENT — NOT part of the benchmark harness.** This is a TiledArray-backend kernel
> modification, env-gated by `SPTC_CE_E_GEMM=1` and **default-off** (unset = byte-identical stock
> backend). It is preserved here for reproducibility only. Do **not** enable it for a
> framework-comparison run. See `docs/HARNESS_VS_EXPERIMENTS.md`.

Env-gated (default off) TiledArray-fork kernel that coalesces **op-488**
(`src/generated_t2_residual.cpp:488`, the ToT×ToT outer-μ̃ contraction forming CSE37,
`…μ̃,Κ;a * …μ̃;a -> …Κ;a,a`) from per-cell `std::function` dispatch into one batched GEMM per
result outer cell — the ce+e analogue of the landed `SPTC_SCALE_GEMM` (op-487, `patches/scale_gemm/`).
This is the concrete "runtime-evaluator work-coalescing / keep-BLAS-fed" piece of the generator/backend
project (`docs/MPQC_MULTIRANK.md`, `docs/MPQC_PROFILE_DEEP.md`).

## What it does
Under owning-ToT (required for multi-rank), op-488 never reaches a GEMM: TA's arena ce+e-GEMM
(`arena_strided_dgemm_ce_e`) is compiled out by the `is_tensor_view_v` guard, so the contraction
fragments into millions of tiny per-cell `std::function` dispatches (the `std::_Function_handler`
self-time). This adds an **owning-cell** variant `arena_strided_dgemm_ce_e_owning` that, per result cell
(m,n), GATHERS L's and R's contracted-μ̃ runs of inner-`a` vectors into contiguous scratch `[K×P]`/`[K×Q]`
and issues one GEMM `C[P×Q] += factor·Lmatᵀ·Rmat` (K=μ̃), with a per-k rank-1 fallback for non-uniform
runs. Installed in the owning builder (`init_inner_tile_op_owning_`) into the existing
`arena_strided_dgemm_ce_e_tile_op_` slot, wired via the existing `set_strided_oprod_op` path, gated by
`ce_e_gemm_enabled()` (`SPTC_CE_E_GEMM`). `[ce-e-fused]` counters mirror `[scale-fused]`.

## Files (against the `patches/scale_gemm/` baseline — i.e. these are the ce+e additions on top of scale-GEMM)
- `arena_einsum.h.{full,patch}` — `ce_e_gemm_enabled()` gate, `arena_strided_dgemm_ce_e_owning` kernel,
  `[ce-e-fused]` counters.
- `cont_engine.h.{full,patch}` — the env-gated install in `init_inner_tile_op_owning_`.
- `gap_microbench.cpp.full` — split L487/L488 timers to isolate op-488's ceiling.
The `.full` files are the complete headers (scale-GEMM + ce+e); the install prefix is
`third_party/tiledarray-cd53bd3-clang/install/include/TiledArray/{tensor,expressions}/`. Rebuild any
owning target (`-DCMAKE_CXX_FLAGS=-DSPTC_OWNING_TOT`) to pick them up.

## Result (RESULTS.md / `docs/scaling-campaign-data/ce_e_gemm.csv`)
- **Correctness:** checksum-exact vs baseline (nnz/sumsq identical, sum to ~13 digits = FP order);
  211435 batched cell-GEMMs, 0 fallbacks. Default (gate off) byte-identical.
- **Single-node win:** C2H6 1.16×@1thr / **1.27×@8thr**; C3H8 1.09×@1thr / **1.19×@8thr** — larger at 8
  threads (the "keep BLAS fed" signature), complementary to `SPTC_SCALE_GEMM`.
- **Does NOT transfer to multi-rank:** C3H8 np4 0.99×, np8 1.03×; C4H10 np8 0.98× — the win decays from
  np1 (single node) to ~neutral by np4-8, because the distributed μ̃ SUMMA splits each rank's GEMM K (the
  same failure mode as scale-GEMM). Confirms the multi-rank scaling term is work **distribution**
  (ProcGrid/layout — the follow-on lever), not per-op dispatch/coalescing.
