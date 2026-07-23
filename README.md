# sequant-ta-repro

A native SeQuant → TiledArray reproduction of MPQC's own closed-shell
CSV-CCSD T1/T2 residual evaluation — no MPQC installation required. This
is the "our own from-scratch implementation" side of a performance-parity
investigation; the ground-truth comparison target (real MPQC, built from
an instrumented fork, plus the real ethane leaf/reference data both sides
were validated against) lives in the sibling repo
[`jianjianh1/mpqc-benchmark`](https://github.com/jianjianh1/mpqc-benchmark).

## What's here

- `src/` — hand-written TiledArray infrastructure (`ta_tensors.h`,
  `ta_builder.h`, `ta_tensor_loader.h`, `coo_loader.h`, `ta_dumper.h`,
  `ta_stage.h`, `steps_csv_loader.h`) plus the actual generated residual
  code: `generated_t1_residual.cpp` / `generated_t2_residual.cpp`
  (pasted verbatim from SeQuant's `TiledArrayGenerator` export backend —
  see `ta_sequant_native_residual.h`'s header comment for provenance and
  exact regeneration steps), driven by `ta_sequant_native_residual_main.cpp`.
- `setup-tiledarray.sh` — clones, builds, and installs TiledArray at the
  exact pinned commit (`84411a6`) this code was validated against.
- `make_bisect_driver.py` — generates a standalone driver for
  `SPTC_CSE_BISECT_N`-style bisection when debugging the derivation
  pipeline's cross-term CSE (see `sequant-fork`).

## Build

```bash
./setup-tiledarray.sh              # ~20-30 min; installs to third_party/tiledarray-84411a6/install
cmake -B build -DCMAKE_BUILD_TYPE=Release .
cmake --build build -j"$(nproc)" --target ta_sequant_native_residual_main
```

## Run against real ethane data

Leaf data + the known-correct reference checksums live in the sibling
`mpqc-benchmark` repo's `traces/checksum-run/` (clone it alongside this
repo, or point at wherever you keep it):

```bash
export SPTC_TRIALS=3 SPTC_WARMUP=1
./build/ta_sequant_native_residual_main \
  ../mpqc-benchmark/traces/checksum-run/sptc_coo_iter1
```

Expected checksums (must match exactly, modulo last-few-ULP float
reassociation noise from CSE restructuring):

| Residual | nnz | sum | sumsq | max_abs |
| --- | --- | --- | --- | --- |
| T1 | 522 | -0.0888799157 | 0.0026454022 | 0.0151662826 |
| T2 | 98598 | 0.2141888066 | 0.0997316723 | 0.0174497557 |

## Regenerating the residual code

`generated_t1_residual.cpp`/`generated_t2_residual.cpp` are not derived
at build time (SeQuant isn't a build dependency of this repo) — they're
pasted output from
[`jianjianh1/sequant-fork`](https://github.com/jianjianh1/sequant-fork)'s
`tests/manual/test_csv_ccsd_derivation.cpp` (branch
`csv-ccsd-tiledarray-generator`), which replicates MPQC's own
`cck.ipp`/`sequant.cpp` derivation pipeline
(biorthogonal transform → `tail_factor` → `density_fit` → `csv_transform`
→ `flatten` → single-term `optimize()`) and exports through
`TiledArrayGenerator`. Regenerate + re-paste here if that pipeline
changes; see `ta_sequant_native_residual.h`'s header comment for the
exact leaf-parameter → `TATensors` field mapping this call site depends
on.

## Key validated performance findings

Living documentation from the investigation that produced this code, so
it doesn't only live in a chat transcript:

- **ArenaTensor storage backend**: switching the tensor-of-tensor
  (CSV/PNO) leaf/intermediate storage from the default TiledArray tile
  type to `TA::Tensor<TA::ArenaTensor<double>>` was required just to make
  the ragged, per-occupied-pair-domain contractions in this workload
  correct/performant at all (see `ta_tensors.h`).
- **MADNESS ThreadPool wait policy**: `Yield` beats the MADNESS-default
  busy-spin AND beats `Sleep(1000us)` (MPQC's own real production
  choice) on combined T1+T2 wall-clock, at every thread count tested —
  set via `SPTC_MAD_WAIT_POLICY=yield` (see
  `ta_sequant_native_residual_main.cpp`).
- **Thread count**: `MAD_NUM_THREADS=10` beats the hardware-concurrency
  default (16 on an 8-core/2-way-SMT box) by ~13-23%. Confirmed via
  profiling that this is a contention effect (fewer threads on the same
  shared spinlock-protected task queue), not a change in the
  scheduling/compute balance.
- **Cross-term common-subexpression elimination**: SeQuant ships a
  previously-unused primitive, `sequant::opt::eliminate_common_subexpressions()`
  (`core/optimize/common_subexpression_elimination.hpp`), that hoists
  shared subexpressions across an entire summand list (not just within
  one term) using the same canonicalization machinery as the runtime
  eval cache. Applying it across T1's 26 and T2's 55 summands is worth
  ~25% (T1) and ~39% (T2) wall-clock. It exposed two real, previously
  latent bugs in the export path (both fixed in
  `test_csv_ccsd_derivation.cpp`, see `sequant-fork`):
  - a CSE-tensor axis-order mislabeling when a later occurrence is a
    pure token-permutation of its defining occurrence,
  - a rank collision where two genuinely different-rank CSE-restructured
    nodes could dedup onto the same exported C++ variable name, causing
    a non-deterministic MADNESS deadlock on Release builds.
- **PaRSEC task backend**: confirmed dramatically slower (~6x) than
  Pthreads for this exact workload/TiledArray build on this hardware —
  all of the above tuning is Pthreads-specific and does not transfer to
  MPQC's real production configuration (PaRSEC by default). See
  `mpqc-benchmark`'s README for why that still makes this comparison
  meaningful.
- **Tile granularity** (`SPTC_TILES_PER_DIM` in `ta_builder.h`, default
  8): swept 2/4/8/16 — coarser (2, 4) timed out, finer (16) was clearly
  worse. The default is already the local optimum; not a further lever.
- **Net gap vs. real MPQC**: after all of the above, real MPQC (PaRSEC,
  its actual production configuration) is still faster than this native
  reproduction by roughly ~2.4x on the real ethane workload. An earlier
  single-data-point reading that looked like a ~0.90x ("we're faster")
  result was traced to MPQC's `eval_level` trace computing a real
  checksum on every intermediate step — genuine extra work absent from
  this repo's own benchmark, which only checksums the final result — and
  was not trusted over the many independently-validated real speedups
  above.
