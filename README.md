# sequant-ta-repro

A native SeQuant → TiledArray reproduction of MPQC's closed-shell CSV-CCSD
T1/T2 residual — **no MPQC installation required**.

The point of this repo is **finer control of what contractions are made**, so
the same contraction sequence can be benchmarked against any tensor-contraction
framework, **without cache/reuse**. Instead of MPQC's runtime tensor-expression
evaluator, we generate an explicit, named `TA::einsum` sequence from SeQuant
(`src/generated_t1_residual.cpp` / `src/generated_t2_residual.cpp`) and run it
directly, so the contraction order, the intermediates, and the tiling are all
things we choose and can inspect, rather than decisions made inside an opaque
evaluator.

The committed sequence is **cache-free**: it is regenerated with `SPTC_NO_CSE=1`
(un-deduped, per-summand, **no cross-term common-subexpression elimination** —
162 and 503 contractions for T1/T2), so nothing is reused across terms. That is
the "without cache/reuse" cost the benchmark exists to measure. The **backend is
stock TiledArray** — the perf-campaign kernel modifications are quarantined and
default-off (`docs/HARNESS_VS_EXPERIMENTS.md`), so the TA numbers are an honest
yardstick. The same SeQuant forest emits other
backends too: a **NumPy einsum** program (`backends/numpy/`, run by
`tools/numpy_runner.py`) comes from the same forest — proving a second framework
is a small additive generator block (the flat sequence is faithful; the CSV/PNO
tensor-of-tensor part needs a generator enhancement — `docs/NUMPY_BACKEND.md`).
MPQC is kept as a no-rebuild cross-check reference (`docs/MPQC_REFERENCE_FLAGS.md`).

The ground-truth comparison target — real MPQC built from an instrumented
fork, plus the ethane leaf/reference data both sides were validated against
— lives in the sibling repo
[`jianjianh1/mpqc-benchmark`](https://github.com/jianjianh1/mpqc-benchmark).
See [`docs/MPQC_COMPARISON.md`](docs/MPQC_COMPARISON.md) for the full parity
analysis and why the tuning knobs below exist, and
[`docs/MPQC_EVALUATION.md`](docs/MPQC_EVALUATION.md) for a stage-by-stage trace
of *how* MPQC evaluates the residual, each stage's repro-match status, and the
verified conclusion on the cold-gap lever (it is the ToT-`einsum` representation,
not tiling or pmap — both empirically ruled out).

## Layout

- `src/generated_t{1,2}_residual.cpp` — the explicit **cache-free** contraction
  sequences (regenerated with `SPTC_NO_CSE=1`, no cross-term reuse), exported
  from SeQuant's `TiledArrayGenerator` and post-processed by
  `tools/postprocess_generated.py`. This is the artifact the repo exists to
  control; **do not hand-edit** (see *Regenerating* below).
- `tools/postprocess_generated.py` — reproducible transform from raw generator
  output to the committed form (arena ToT type → `ArrayToT` alias + header).
- `backends/numpy/` + `tools/numpy_runner.py` — the *same* sequence emitted for
  NumPy einsum from the same SeQuant forest (second-backend proof; see
  `docs/NUMPY_BACKEND.md` for the runnable-status caveat).
- `experiments/` + `patches/` — quarantined perf-campaign backend modifications,
  all default-off (`docs/HARNESS_VS_EXPERIMENTS.md`); NOT the benchmark.
- `src/ta_builder.h` — the tiling-control core: builds the flat and
  tensor-of-tensor (CSV/PNO) arrays and decides their tile boundaries. This
  is where the `SPTC_*` tiling knobs live.
- `src/ta_tensors.h` — the leaf tensor types, including the `SPTC_OWNING_TOT`
  compile-time switch for the tensor-of-tensor inner-tile layout.
- `src/ta_sequant_native_residual.h` — the hand-maintained
  leaf → `TATensors` field mapping the generated code binds against, plus
  `permute_ij_*`. Must be re-synced whenever the generated code is
  regenerated.
- `src/{coo_loader,ta_tensor_loader,ta_stage,ta_dumper}.h` — COO leaf
  loading, per-molecule load orchestration, MADNESS init/staging, and the
  reference checksum.
- `src/ta_sequant_native_residual_main.cpp` — the driver.
- `docs/MPQC_COMPARISON.md` — the performance-parity writeup.
- `docs/MPQC_EVALUATION.md` — end-to-end trace of MPQC's CSV-CCSD residual
  evaluation + "how to match it" assessment (the cold-gap lever analysis).
- `docs/CONTRACTION_IR.md` — CTIR, a human-readable Contraction IR that shows
  what the einsum C++ hides (per-cell cost / CELL-BOUND, t-indep vs t-dep +
  persistent, aux-Κ batchability) and why the einsum can't describe MPQC's
  runtime computation. Samples in `docs/scaling-campaign-data/*.ctir`.
- `docs/LAYERED_IR.md` — design/feasibility study for a stack of layered IRs
  that progressively lower the residual from the MPQC/SeQuant equation to
  einsum sequences (MLIR-style); shows the pipeline is already an informal
  lowering, and that the MPQC-vs-repro difference is a single scheduling pass.
- `tools/` — optional diagnostics/benchmarks (off by default; see
  `tools/README.md`).

## The benchmark build

There is **one** canonical benchmark build: **stock TiledArray, owning
tensor-of-tensor tiles, every experiment gate unset.** This is the honest,
unmodified backend — no custom kernels, nothing from `docs/HARNESS_VS_EXPERIMENTS.md`
§B is compiled in or firing. The residual links against a **clang-built**
TiledArray, so it must itself be built with clang (matching ABI).

```bash
./setup-tiledarray.sh    # clones + builds TiledArray cd53bd3 (clang-21, OpenBLAS)
                         # into third_party/tiledarray-cd53bd3-clang/install; ~20-30 min
# Canonical build: owning ToT (-DSPTC_OWNING_TOT), Release, no gates.
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER=clang++-21 -DCMAKE_C_COMPILER=clang-21 \
      -DCMAKE_CXX_FLAGS=-DSPTC_OWNING_TOT .
cmake --build build -j"$(nproc)" --target ta_sequant_native_residual_main
```

Owning ToT is the stock, thread-safe (`MAD_NUM_THREADS>1`), multi-rank-capable
build the committed checksums use. (Omitting `-DSPTC_OWNING_TOT` gives the
arena-view variant — also stock TA, but single-thread-only on cd53bd3; see
`docs/HARNESS_VS_EXPERIMENTS.md`.) If TiledArray is installed elsewhere, pass
`-DTA_INSTALL_DIR=/path/to/install`. **Do not** pass any `SPTC_*_GEMM` /
`SPTC_AUX_TARGET_SIZE` / `SPTC_COMPACT_COEFFS` gate for a benchmark run — those
are quarantined experiments, not the harness.

## Run + validate

Leaf data and the known-correct reference checksums live in the sibling
`mpqc-benchmark` repo's `traces/checksum-run/`:

```bash
SPTC_COARSE_OCC=9 SPTC_OCC_TILE=2 SPTC_COARSE_PAD=0 SPTC_TILES_PER_DIM=4 \
SPTC_MAD_WAIT_POLICY=yield MAD_NUM_THREADS=8 \
  taskset -c 0-7 ./build/ta_sequant_native_residual_main \
  ../mpqc-benchmark/traces/checksum-run/sptc_coo_iter1
```

> **Note:** the canonical `./build/` above is the **owning** build (`-DSPTC_OWNING_TOT`), which is
> thread-safe at `MAD_NUM_THREADS>1` and is what the committed checksums use. If you instead build the
> arena-view variant (omit `-DSPTC_OWNING_TOT`), run it with `MAD_NUM_THREADS=1` — its ToT inner cells
> are freed cross-thread by MADNESS lazy deletion and segfault above one thread.

The run must reproduce these checksums exactly, modulo last-few-ULP float
reassociation noise (the cache-free sequence recomputes shared intermediates,
so it is slower than a CSE'd/cached evaluator by design — that recompute cost
is what "without cache/reuse" means and what this benchmark measures):

| Residual | nnz | sum | sumsq | max_abs |
| --- | --- | --- | --- | --- |
| T1 | 522 | -0.0888799157 | 0.0026454022 | 0.0151662826 |
| T2 | 98598 | 0.2141888066 | 0.0997316723 | 0.0174497557 |

(`taskset -c 0-7` pins to 8 physical cores, avoiding SMT siblings; adjust
the core list and `MAD_NUM_THREADS` to your machine.)

## Controlling the contractions and tiling

The primary feature. Set via environment variables at run time:

**Tiling** (`ta_builder.h`):
- `SPTC_COARSE_OCC=<N>` — occupied-extent threshold above which the
  tensor-of-tensor outer occupied-pair dimension is coarse-tiled instead of
  forced to size-1 pair-key tiles. `0` (default) = size-1 tiling.
- `SPTC_OCC_TILE=<N>` — occupied pairs per coarse tile when `SPTC_COARSE_OCC`
  is active.
- `SPTC_COARSE_PAD=0|1` — pad ragged inner (PNO) extents to a common size
  within a coarse tile (`1`) vs. keep them ragged (`0`, default).
- `SPTC_TILES_PER_DIM=<N>` — target tile count per dimension for ToT outer
  dims (and flat arrays, unless overridden below).
- `SPTC_FLAT_TILES_PER_DIM=<N>` — override the target tile count for **flat**
  arrays only, independent of `SPTC_TILES_PER_DIM`.

**Scheduling / measurement** (`ta_sequant_native_residual_main.cpp`):
- `SPTC_MAD_WAIT_POLICY=busy|yield|sleep` — MADNESS threadpool idle policy
  (`yield` is the validated best; `sleep` is MPQC's production choice).
- `MAD_NUM_THREADS=<N>` — MADNESS compute threads (best ≈ physical core count
  when pinned).
- `SPTC_TRIALS=<N>` / `SPTC_WARMUP=<N>` — timing trials and warmup iterations.

**Compile-time** (`ta_tensors.h`):
- `-DSPTC_OWNING_TOT` — use owning `TA::Tensor<double>` inner cells instead
  of the default arena-view `TA::ArenaTensor<double>`. Numerically identical;
  required for `MAD_NUM_THREADS>1` on TA cd53bd3 (see `docs/MPQC_COMPARISON.md`
  §8).

## Regenerating the contraction code

The generated residuals are **not** derived at build time (SeQuant is not a
build dependency here) — they are pasted output from
[`jianjianh1/sequant-fork`](https://github.com/jianjianh1/sequant-fork)'s
`tests/manual/test_csv_ccsd_derivation.cpp` (branch
`csv-ccsd-tiledarray-generator`), which replicates MPQC's own derivation
pipeline (biorthogonal transform → `tail_factor` → `density_fit` →
`csv_transform` → `flatten` → single-term `optimize()`) and exports through
`TiledArrayGenerator`.

To regenerate the **cache-free** (default) sequence:

```bash
# 1. Derive + export un-deduped (no cross-term CSE). Writes /tmp/claude-ta-generator-test/generated_R{1,2}.cpp
SPTC_NO_CSE=1 MAD_NUM_THREADS=1 <build>/tests/manual/ta_generator_cc_test

# 2. Rewrite the arena ToT type -> the ArrayToT alias + add the provenance header (reproducible; no hand-edit).
python3 tools/postprocess_generated.py /tmp/claude-ta-generator-test/generated_R1.cpp 1 <date> > src/generated_t1_residual.cpp
python3 tools/postprocess_generated.py /tmp/claude-ta-generator-test/generated_R2.cpp 2 <date> > src/generated_t2_residual.cpp
```

Then **re-sync the leaf-parameter → `TATensors` field mapping** in
`src/ta_sequant_native_residual.h`: the parameter *order* changes on
regeneration (the NAME↔field mapping does not). Run the post-processor with
`--print-order` to see the new order, and update the two forward declarations
and two `whole_t{1,2}_residual(...)` wrapper calls to match. Verify by rebuilding
and reproducing the checksums above — the sequence changes, the residual must not.

(Omit `SPTC_NO_CSE=1` to regenerate the CSE'd/reuse variant instead — a
different, deduped sequence with the same result, kept available via this toggle
for cache-vs-no-cache comparison. The **default committed sequence is
cache-free**.)
