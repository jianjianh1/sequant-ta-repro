# tools/ — optional diagnostics and benchmarks

Investigation utilities from the MPQC performance-parity study. None are on
the core build path; the residual driver (`ta_sequant_native_residual_main`)
builds and runs without them. Build the C++ ones with
`cmake -B build -DSPTC_BUILD_TOOLS=ON .`.

- **`emit_split.py`** — SSA-splits `src/generated_t2_residual.cpp` into a
  t-independent "precompute once" block plus a t-dependent residual (MPQC's
  volatile/persistent cache split), writing `tools/gen_split/t2_split.{cpp,h}`.
  Run: `python3 tools/emit_split.py`.
- **`gen_split/` + the `ta_warm_t2` target** — the generated warm-loop T2
  benchmark: precomputes the t-independent DF/CSV block once, then times the
  t-dependent residual across trials (the "warm iteration" analog of MPQC's
  cached steady state). Run like `ta_sequant_native_residual_main` (same env
  knobs + `<trace_dir>`).
- **`sptc_traced_einsum.h`** — a drop-in `sptc::einsum` that fences, times,
  and checksums each contraction to a CSV (`SPTC_TRACE_OPS_PATH`), for
  per-operation comparison against MPQC's trace. Compile a copy of the
  generated residual with `TA::einsum` textually replaced by `sptc::einsum`.
- **`make_bisect_driver.py`** — generates a standalone `SPTC_CSE_BISECT_N`
  driver to bisect cross-term CSE when debugging the SeQuant derivation.
- **`gap_microbench.cpp` + the `gap_microbench` target** — achievable-speedup
  ceiling for the cold CSE37 hotspot (`generated_t2_residual.cpp:487-488`): the
  giant μ̃Κ block done as a hand per-pair BLAS GEMM vs `TA::einsum` on identical
  data (`-DSPTC_OWNING_TOT`).
  Reads `MAD_NUM_THREADS`, `SPTC_TRIALS`.

See `docs/HARNESS_VS_EXPERIMENTS.md` for their non-canonical status.
