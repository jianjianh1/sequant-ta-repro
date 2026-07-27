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
export SPTC_MAD_WAIT_POLICY=yield MAD_NUM_THREADS=8 SPTC_TRIALS=3 SPTC_WARMUP=1
taskset -c 0-7 ./build/ta_sequant_native_residual_main \
  ../mpqc-benchmark/traces/checksum-run/sptc_coo_iter1
```

(`taskset -c 0-7` pins to this box's 8 physical cores, avoiding SMT
siblings — see "Key validated performance findings" below for why this
combination, not just `MAD_NUM_THREADS`, is the current best-known
config. Adjust the core list/thread count to match your own machine's
physical core count.)

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
- **CPU affinity pinning (supersedes the thread-count finding above)**:
  a follow-up 8-way sweep of `{unpinned, taskset -c 0-7} x
  MAD_NUM_THREADS in {7,8,9,10}` found that pinning the whole process to
  the 8 physical cores (`taskset -c 0-7`, avoiding SMT siblings) beats
  every unpinned configuration at every thread count, and — once
  pinned — the OPTIMUM FLIPS from more threads to fewer: `MAD_NUM_THREADS=8`
  (exactly the physical core count) now beats 9 and 10, whereas unpinned,
  10 beat 7-9. Interpretation: unpinned, extra threads were compensating
  for OS scheduling/migration noise; pinning removes that noise, so
  over-provisioning threads past the physical core count just re-adds
  contention. Net win over the previous best (unpinned, 10 threads):
  T1 ~24% faster (1.154s → 0.872s), T2 ~20% faster (10.047s → 8.082s).
  Motivated directly by a fresh `perf` profile showing T1's wall-time is
  ~50%+ kernel-side `sched_yield`/scheduling machinery (almost no real
  compute visible in the top functions) — proportionally *worse* than
  T2's ~20-25% scheduling share (T2's profile clearly shows real compute,
  e.g. `fused_scale_t_x_tot_inplace` on `ArenaTensor` tiles, at ~21%
  self-time). That asymmetry is itself still unexplained — it does NOT
  simply mean T2 is more efficient than T1 relative to real MPQC (see the
  "Net gap" bullet below, where T2's ratio is still worse) — confirming
  it would need a comparably fresh, per-residual profile of real MPQC's
  own PaRSEC execution, which hasn't been done.
- **Real MPQC tiling for tensor-of-tensor arrays — tried, hit a genuine
  TiledArray bug, not adopted**: `ta_builder.h`'s `real_tiling_sidecar`
  mechanism (opt-in, loads MPQC's own coarser tile boundaries instead of
  forcing pair-key dims to tile size 1) was wired up end-to-end and
  tested on real ethane data. Result: a heap-buffer-overflow (confirmed
  via AddressSanitizer) inside TiledArray's `SparseShape` destruction
  path, triggered by MADNESS's asynchronous cross-thread lazy-deletion
  whenever a tensor-of-tensor array's outer tiling spans more than one
  occupied pair per tile — a different, previously-undiscovered bug from
  the one the tile-size-1 default already guards against. Looks like a
  genuine TiledArray/MADNESS-side issue on the pinned `84411a6` commit,
  not something fixable in this repo — see `ta_builder.h`'s own comment
  for the full diagnosis. Don't re-attempt without first confirming a
  newer TiledArray commit fixes it.
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
- **Net gap vs. real MPQC**: a correctness-gated, trace-independent,
  fresh side-by-side measurement (both sides on OpenBLAS, single-rank,
  their own real best settings — MPQC's PaRSEC production defaults vs.
  this repo's pinned/`MAD_NUM_THREADS=8` config above) puts real MPQC
  ahead by **~1.89x on T1, ~3.59x on T2, ~3.30x combined** (T1: 0.461s
  MPQC vs. 0.872s here; T2: 2.254s MPQC vs. 8.082s here). This supersedes
  an earlier ~2.4x figure that was only a scaled *estimate* from
  older, cross-phase data — the first genuinely fresh, direct
  measurement (before the affinity-pinning win above) actually found a
  *larger* gap (~2.50x/~4.46x/~4.13x), underscoring that this number
  should be re-measured directly rather than assumed stable across
  changes. Separately, an even earlier single-data-point reading that
  looked like a ~0.90x ("we're faster") result was traced to MPQC's
  `eval_level` trace computing a real checksum on every intermediate
  step — genuine extra work absent from this repo's own benchmark, which
  only checksums the final result — and was correctly not trusted over
  independently-validated real speedups.
