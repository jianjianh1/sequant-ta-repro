# experiments/ — quarantined perf-campaign artifacts (NOT the benchmark)

Everything here is a **default-off** modification from the MPQC perf-parity campaign, kept for
reproducibility. None of it is on the canonical benchmark path (`README.md` → *The benchmark build*), which
uses a **stock** TiledArray backend with all of these disabled. A modified backend can't be an honest
yardstick for comparing other frameworks — that is the whole reason these are quarantined here rather than
in `src/`. Full rationale + the gate inventory: `docs/HARNESS_VS_EXPERIMENTS.md`.

## In this directory
- `ta_runtime_eval_main.cpp` — the `sequant::evaluate()` runtime-evaluator driver (MPQC's runtime tree-walk
  path). Built only with CMake `-DSPTC_BUILD_RUNTIME_EVAL=ON` (default OFF) and a SeQuant install. It is a
  *control* (measured ~1.2× slower than the explicit sequence), the opposite of this repo's goal of an
  explicit, controllable contraction sequence. Patch: `../patches/runtime_eval/`.

## Elsewhere in the tree (also quarantined, cross-referenced here)
- `../patches/scale_gemm/` — `SPTC_SCALE_GEMM` TA kernel (op-487 μ̃Κ half-transform → batched GEMM).
- `../patches/ce_e_gemm/` — `SPTC_CE_E_GEMM` TA kernel (op-488 outer-μ̃ → batched cell-GEMM).
- `../src/aux_k_batching.h` + `../src/generated_t2_residual_auxbatch.cpp` + the `ta_auxbatch_main` target —
  `SPTC_AUX_TARGET_SIZE` aux-Κ batching (needs the fork trange-assertion relaxation).
- `../tools/gap_microbench.cpp` — the op-487/488 GEMM-ceiling microbenchmark (`SPTC_BUILD_TOOLS=ON`).
- Git history — the measurements that motivated and evaluated these experiments.

These are findings, not the product. The product is the cache-free, controllable, multi-backend
contraction-sequence benchmark in `src/`, `tools/postprocess_generated.py`, and `tools/numpy_runner.py`.
