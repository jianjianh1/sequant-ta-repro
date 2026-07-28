# sequant-ta-repro

A native SeQuant → TiledArray reproduction of MPQC's closed-shell CSV-CCSD
T1/T2 residual — **no MPQC installation required**.

The point of this repo is **finer control of what contractions are made**.
Instead of MPQC's runtime tensor-expression evaluator, we generate an
explicit, named `TA::einsum` sequence from SeQuant
(`src/generated_t1_residual.cpp` / `src/generated_t2_residual.cpp` — 119 and
252 named contractions) and run it directly, so the contraction order, the
intermediates, and the tiling are all things we choose and can inspect,
rather than decisions made inside an opaque evaluator.

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

- `src/generated_t{1,2}_residual.cpp` — the explicit contraction sequences,
  pasted verbatim from SeQuant's `TiledArrayGenerator` export backend. This
  is the artifact the repo exists to control; **do not hand-edit** (see
  *Regenerating* below).
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
- `tools/` — optional diagnostics/benchmarks (off by default; see
  `tools/README.md`).

## Build

The residual links against a **clang-built** TiledArray, so it must itself
be built with clang (matching ABI).

```bash
./setup-tiledarray.sh    # clones + builds TiledArray cd53bd3 (clang-21, OpenBLAS)
                         # into third_party/tiledarray-cd53bd3-clang/install; ~20-30 min
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER=clang++-21 -DCMAKE_C_COMPILER=clang-21 .
cmake --build build -j"$(nproc)" --target ta_sequant_native_residual_main
```

If TiledArray is installed elsewhere, pass `-DTA_INSTALL_DIR=/path/to/install`.

## Run + validate

Leaf data and the known-correct reference checksums live in the sibling
`mpqc-benchmark` repo's `traces/checksum-run/`:

```bash
SPTC_COARSE_OCC=9 SPTC_OCC_TILE=2 SPTC_COARSE_PAD=0 SPTC_TILES_PER_DIM=4 \
SPTC_MAD_WAIT_POLICY=yield MAD_NUM_THREADS=8 \
  taskset -c 0-7 ./build/ta_sequant_native_residual_main \
  ../mpqc-benchmark/traces/checksum-run/sptc_coo_iter1
```

The run must reproduce these checksums exactly, modulo last-few-ULP float
reassociation noise:

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

To regenerate: run that test, paste its output over
`src/generated_t{1,2}_residual.cpp`, and **re-sync the leaf-parameter →
`TATensors` field mapping** in `src/ta_sequant_native_residual.h` (its header
comment documents the exact mapping the generated code depends on).
