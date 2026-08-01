# scale-GEMM patch — measured results

Env-gated by `SPTC_SCALE_GEMM=1` (default off = byte-identical baseline). Arena-only (requires
`is_tensor_view` inner cells); the shipped multi-rank build is owning-ToT, so this kernel does not run
there. Checksum gate: whole-T2 nnz (C2H6=150301, C3H8=261914), sumsq/max_abs; `sum` differs at ~1e-13
(GEMM vs sequential-AXPY summation order).

## Single-node (node3, clang/OpenBLAS arena, C2H6 cc-pVTZ cold T2)

| config | wall | note |
|---|---|---|
| 1 thread, baseline | 30.60 s | 358.8 M per-cell scalar AXPYs |
| 1 thread, SPTC_SCALE_GEMM=1 | **20.01 s** | 1.53×; scale AXPY calls → 0 |
| 8 threads, baseline | 5.94 s | already beats MPQC's 7.8 s |
| 8 threads, SPTC_SCALE_GEMM=1 | **4.94 s** | 1.20×; 1.58× faster than MPQC |

Scaling (1 thread): C3H8 187.6 → 129.2 s (1.45×). Op-level ceiling (gap_microbench, isolated μ̃Κ
block): C2H6 7.4×, C3H8 7.3×, checksum-matched.

## Multi-rank — does NOT help (do not enable multi-rank)

C3H8 cold, np=2: baseline 50.8 s vs `SPTC_SCALE_GEMM=1` 70.8 s (**1.4× slower**, checksum-exact). The
distributed SUMMA splits contracted μ̃ into small K-panels per `strided_oprod_op` call → tiny GEMM
loses to per-cell AXPY. Single-node-only lever. See `docs/MPQC_MULTIRANK.md`.

## What / how
The μ̃Κ half-transform `I(i,i,μ̃,Κ;a)=Σ_μ̃ g(μ̃,μ̃,Κ)·C(i,i,μ̃;a)` (generated_t2:487) + ~50 cousins have
the PNO index `a` as a ToT inner spectator, so TA dispatched them as per-cell scalar AXPY, not a GEMM.
The kernel `arena_strided_scale` gathers each pair's μ̃-run into `Rmat[K×Q]` and issues one GEMM
`tmp[M×Q]=L[M×K]·Rmat` — MPQC's per-pair GEMM form. See `README.md` for apply/rebuild.
Full analysis: `docs/MPQC_SINGLE_THREAD.md`.
