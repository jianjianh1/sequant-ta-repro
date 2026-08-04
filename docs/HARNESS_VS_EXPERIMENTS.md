# Harness vs experiments — what is the benchmark, and what is not

This repo's purpose is a **framework-agnostic, controllable, cache-free contraction-sequence benchmark**:
drive the explicit CSV-CCSD residual sequence (`src/generated_t{1,2}_residual.cpp`) against a tensor
framework, without cache/reuse, with finer control than MPQC's runtime evaluator (`README.md`).

The perf-parity campaign (docs/MPQC_*.md) added a number of **TiledArray/SeQuant modifications** to chase
MPQC's runtime speed. Those are legitimate findings, but they are **not part of the benchmark** — a
hand-optimized backend can't be an honest yardstick for other frameworks. This document draws the line:
which knobs *are* the benchmark (they shape/measure the sequence without modifying the backend), and which
are **quarantined experiments** (they modify the backend or the evaluator; default-off; preserved under
`patches/` + git history for reproducibility, not on the default path).

**Rule:** the canonical benchmark build has **every experiment gate below unset** and links **no custom
kernel**. See `README.md` → *The benchmark build*.

---

## A. Benchmark harness controls — this IS the benchmark

These shape or measure the contraction sequence on a **stock** backend. They do not modify TiledArray or
SeQuant's evaluator. Using them is the intended workflow.

| Knob | Where | What it controls |
| --- | --- | --- |
| `SPTC_NO_CSE` | regeneration (SeQuant `test_csv_ccsd_derivation.cpp`) | **The cache-free lever.** Exports the residual un-deduped, per-summand, no cross-term CSE. The default committed sequence is generated with this ON (`tools/postprocess_generated.py`). |
| `-DSPTC_OWNING_TOT` | compile (`ta_tensors.h`) | Owning `TA::Tensor<TA::Tensor<double>>` inner tiles (canonical) vs arena views. Numerically identical; owning is the stock, thread-safe, multi-rank build. |
| `SPTC_COARSE_OCC`, `SPTC_OCC_TILE`, `SPTC_COARSE_PAD`, `SPTC_TILES_PER_DIM`, `SPTC_FLAT_TILES_PER_DIM` | run (`ta_builder.h`) | Tiling of the ToT outer / flat dims — the "finer control" feature. |
| `SPTC_MAD_WAIT_POLICY`, `SPTC_MAD_WAIT_SLEEP_US`, `MAD_NUM_THREADS` | run | MADNESS threadpool idle policy / thread count. |
| `SPTC_TRIALS`, `SPTC_WARMUP` | run | Timing trials / warmup. |
| `SPTC_SPARSE_THRESHOLD` | run (`ta_builder.h`) | Sparse-shape screening threshold. |

## B. Quarantined experiments — NOT the benchmark (default-off)

These **modify the TiledArray backend or SeQuant's runtime evaluator**. They are the campaign's levers.
Each is env- or CMake-gated OFF by default; with the gate unset the build is byte-identical to stock. They
live under `patches/` (full headers + `.patch`) so they can be re-applied to the gitignored TA install, and
their results are in `docs/scaling-campaign-data/`. **Do not enable them for a framework-comparison run** —
they make the TA backend non-representative.

| Gate | Kind | What it modifies | Preserved in | Findings |
| --- | --- | --- | --- | --- |
| `SPTC_SCALE_GEMM` | env, TA kernel | op-487 μ̃Κ half-transform → one batched GEMM per pair | `patches/scale_gemm/` | 1.53×@1thr single-node; does NOT transfer multi-rank |
| `SPTC_CE_E_GEMM` | env, TA kernel | op-488 outer-μ̃ ToT×ToT → batched cell-GEMM | `patches/ce_e_gemm/` | 1.27×@8thr single-node; does NOT transfer multi-rank |
| `SPTC_COMPACT_COEFFS` | env, TA data layout | MPQC `compact_csv_coeffs` port (single-page arena slabs) | (arena path) | enables the strided-DGEMM fast path |
| `SPTC_AUX_TARGET_SIZE` | env, sequence | aux-Κ batching of the DF half-transform | `src/aux_k_batching.h`, `src/generated_t2_residual_auxbatch.cpp`, `ta_auxbatch_main` | cuts peak RSS 43-63%; needs the fork trange-assertion relaxation |
| `SPTC_BUILD_RUNTIME_EVAL`, `SPTC_NO_CACHE` | CMake + env, evaluator | builds the `sequant::evaluate()` runtime T2 driver (MPQC's runtime path) instead of the static sequence; `SPTC_NO_CACHE` disables its CacheManager | `patches/runtime_eval/`, `src/ta_runtime_eval_main.cpp` | the port is 1.2× slower — a control, not a win |
| `SPTC_CYCLIC_PMAP`, `SPTC_REPRO_NOFENCE`, `SPTC_REPRO_EINSUM` | env, distribution | multi-rank layout / fence experiments | (in driver + docs) | distribution levers, all refuted |
| `SPTC_PROD_TRACE`, `SPTC_EVAL_TRACE`, `SPTC_TRACE_OPS_PATH`, `SPTC_DUMP_PMAP` | env, diagnostics | per-op / pmap tracing (measurement overhead) | (in driver / `tools/`) | profiling only |

*Derivation-side experiment gates* (in the SeQuant test, affect regeneration only, not the shipped
sequence): `SPTC_PROTO_EXTENT`, `SPTC_NREPLAY`, `SPTC_OPT_MEMSIZE`, `SPTC_CSE_BISECT_N`. *Misc validation
gates* (driver): `SPTC_SYMMETRIZE_R2`, `SPTC_NO_SYMM`, `SPTC_CHECK_SYM`, `SPTC_CLONE_LEAF`.

## The one unavoidable fork carry

The TiledArray install is a fork (`third_party/tiledarray-cd53bd3-clang`) that relaxes one `dist_eval`
trange-equality assertion — needed only by aux-K batching (§B, `SPTC_AUX_TARGET_SIZE`). It is not reached on
the stock benchmark path and does not change any residual result. Everything else in §B is additive and
gated off.

## See also
- `README.md` — the canonical benchmark build + run.
- `patches/{scale_gemm,ce_e_gemm,runtime_eval}/README.md` — the quarantined kernels, with reproduction.
- `docs/MPQC_REFERENCE_FLAGS.md` — MPQC as a no-rebuild cross-check reference.
- `docs/scaling-campaign-data/` — the campaign measurements behind §B.
