# sequant-ta-repro

A native SeQuant → TiledArray reproduction of MPQC's closed-shell CSV-CCSD
T1/T2 residual, without an MPQC installation.

The repository turns SeQuant's symbolic residual into an explicit named
`TA::einsum` sequence. This exposes contraction order, intermediates, tiling,
and execution controls that MPQC's runtime evaluator normally owns. The
committed sequence was generated with `SPTC_NO_CSE=1`: it is occurrence
preserving and performs no cross-term common-subexpression reuse. It contains
162 T1 and 503 T2 binary contractions across the native 26 R1 and 55 R2 terms.

## What is canonical

The benchmark configuration is:

- stock TiledArray at the pinned `cd53bd3` fork commit;
- clang 21, Release mode, OpenBLAS;
- owning tensor-of-tensor inner tiles (`-DSPTC_OWNING_TOT`);
- all experimental backend/kernel gates unset.

Backend modifications and evaluator experiments live under `patches/` and
`experiments/`. They are retained as implementation evidence but are not part
of a framework comparison. See
[`docs/HARNESS_VS_EXPERIMENTS.md`](docs/HARNESS_VS_EXPERIMENTS.md).

## Build

```bash
./setup-tiledarray.sh

cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=clang++-21 \
  -DCMAKE_C_COMPILER=clang-21 \
  -DCMAKE_CXX_FLAGS=-DSPTC_OWNING_TOT .
cmake --build build -j"$(nproc)" --target ta_sequant_native_residual_main
```

The setup script installs TiledArray under
`third_party/tiledarray-cd53bd3-clang/install`. To use another intentional
installation, pass `-DTA_INSTALL_DIR=/path/to/install`.

## Run and validate

The ethane leaf tensors and reference checksums live in the sibling
`mpqc-benchmark` repository:

```bash
SPTC_COARSE_OCC=9 SPTC_OCC_TILE=2 SPTC_COARSE_PAD=0 \
SPTC_TILES_PER_DIM=4 SPTC_MAD_WAIT_POLICY=yield MAD_NUM_THREADS=8 \
taskset -c 0-7 ./build/ta_sequant_native_residual_main \
  ../mpqc-benchmark/traces/checksum-run/sptc_coo_iter1
```

Expected checksums, allowing only last-ULP reassociation noise:

| Residual | nnz | sum | sumsq | max_abs |
| --- | ---: | ---: | ---: | ---: |
| T1 | 522 | -0.0888799157 | 0.0026454022 | 0.0151662826 |
| T2 | 98598 | 0.2141888066 | 0.0997316723 | 0.0174497557 |

Owning ToT is the supported threaded and multi-rank build. The arena-view
variant, produced by omitting `-DSPTC_OWNING_TOT`, is a single-thread diagnostic
for this TiledArray commit.

## Controls

Runtime tiling controls in `src/ta_builder.h`:

- `SPTC_COARSE_OCC`, `SPTC_OCC_TILE`, `SPTC_COARSE_PAD`
- `SPTC_TILES_PER_DIM`, `SPTC_FLAT_TILES_PER_DIM`
- `SPTC_SPARSE_THRESHOLD`

Scheduling and measurement controls:

- `SPTC_MAD_WAIT_POLICY`, `SPTC_MAD_WAIT_SLEEP_US`
- `MAD_NUM_THREADS`
- `SPTC_TRIALS`, `SPTC_WARMUP`

Do not enable `SPTC_*_GEMM`, `SPTC_AUX_TARGET_SIZE`, or
`SPTC_COMPACT_COEFFS` for a canonical benchmark. Those flags activate
quarantined experiments.

## Layout

- `src/generated_t{1,2}_residual.cpp`: generated cache-free sequences; do not
  hand-edit
- `src/ta_sequant_native_residual.h`: leaf-to-field binding and wrappers
- `src/ta_builder.h`, `src/ta_tensors.h`: tensor construction and tiling
- `src/{coo_loader,ta_tensor_loader,ta_stage,ta_dumper}.h`: input and checksums
- `backends/numpy/`, `tools/numpy_runner.py`: second-backend export scaffold
- `docs/CONTRACTION_IR.md`, `docs/LAYERED_IR.md`: representation design
- `docs/MPQC_REFERENCE_FLAGS.md`: MPQC cross-check configuration
- `docs/MPQC_FAIR_COMPARISON.md`: current comparison contract
- `patches/`, `experiments/`: non-canonical experiments

## Regenerate the sequence

Generation is performed in the sibling `sequant-fork` worktree from
`tests/manual/test_csv_ccsd_derivation.cpp`:

```bash
SPTC_NO_CSE=1 MAD_NUM_THREADS=1 \
  /path/to/sequant-build/tests/manual/ta_generator_cc_test

python3 tools/postprocess_generated.py \
  /tmp/claude-ta-generator-test/generated_R1.cpp 1 <date> \
  > src/generated_t1_residual.cpp
python3 tools/postprocess_generated.py \
  /tmp/claude-ta-generator-test/generated_R2.cpp 2 <date> \
  > src/generated_t2_residual.cpp
```

After regeneration, use `tools/postprocess_generated.py --print-order` to
re-sync the two parameter lists and wrapper calls in
`src/ta_sequant_native_residual.h`, then rebuild and reproduce both checksums.

## Scope

The NumPy backend proves that the same SeQuant forest can target another
einsum API, but the emitted CSV/PNO program still needs per-index metadata and
ragged tensor-of-tensor lowering before it can run end to end. See
[`docs/NUMPY_BACKEND.md`](docs/NUMPY_BACKEND.md). MPQC remains a correctness
and native-runtime reference; `mpqc-benchmark` owns its authoritative 81-term
no-cache measurements.
