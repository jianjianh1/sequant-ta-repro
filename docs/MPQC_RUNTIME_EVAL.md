# Reproducing MPQC's runtime evaluator (`sequant::evaluate`) — it works, and it's slower

*2026-08-01. Plan `pure-discovering-book` ("do the reproduce", user chose "Build the port anyway").
Deliverable: `src/ta_runtime_eval_main.cpp` — compute the CSV-CCSD T2 residual by walking the L2
`EvalNode` forest with SeQuant's **runtime evaluator** (`sequant::evaluate`, the way MPQC's `cck.ipp`
does), instead of the repo's STATIC generated einsum sequence (`src/generated_t2_residual.cpp`). Built as
a definitive control artifact; predicted equal-or-worse performance.*

## TL;DR / verdict

**The port works.** It derives the equations, walks the forest, and produces a T2 residual that matches
the static path to **0.2 % in ‖R‖², 0.14 % in max|R|, with exact nnz** on C2H6. And it is **slower**:
single-rank, single-thread, **50.1 s (runtime) vs 41.0 s (static) = 1.22× slower**. This confirms the
plan's honest prediction (equal-or-worse): both paths funnel every contraction through the same
`TA::einsum`, and the runtime evaluator only *adds* per-node `Result`-wrapping + `wait_for_lazy_cleanup`
drains on top. The remaining sub-percent value gap is a known ToT-coefficient layout detail (below), not a
structural error.

## What it does (verified on C2H6_coo, single rank, `MAD_NUM_THREADS=1`)

1. **Build/deps.** Unified SeQuant (`SEQUANT_TILEDARRAY=ON`, clang++-21, against the repo's
   `tiledarray-cd53bd3-clang/install`) installed to `/users/jianjian/sequant-fork/install-eval`. Gated
   CMake target `ta_runtime_eval_main` (`-DSPTC_BUILD_RUNTIME_EVAL=ON`) resolves SeQuant's transitive deps
   by prepending `install-eval` + `build-eval/_deps/{eigen3,range-v3,libperm,polymorphic_variant}-build`
   and `find_package(Eigen3 CONFIG)` **before** `find_package(SeQuant)`.
2. **Derivation reproduces verbatim** (`test_csv_ccsd_derivation.cpp:265-531`) → 55 summands.
3. **Forest + cache + leaf yielder** all build; every leaf maps.
4. **Evaluate + symmetrize + checksum**, mirroring `cck.ipp` (`evaluate` per summand, sum, then
   `R2("i,j;a,b") = 0.5(R2("i,j;a,b") + R2("j,i;b,a"))`).

## Correctness (raw / pre-symmetrization residual — the factorization-invariant quantity)

| C2H6_coo, 1-thread | nnz | Σ Rᵢ | ‖R‖² | max\|R\| |
|---|---|---|---|---|
| static (`ta_sequant_native_residual`) | 150301 | -6.772e-4 | 0.117959 | 0.0170234 |
| **runtime (`ta_runtime_eval`)** | **150301** | -1.426e-3 | **0.118221** | **0.0170474** |
| rel. diff | exact | 2.1× | **+0.22 %** | **+0.14 %** |

Same picture on **C3H8** (the port generalises): nnz 261914 (exact), ‖R‖² 0.170051 vs static 0.169723
(+0.19 %), max|R| 0.0189441 vs 0.0189115 (+0.17 %), Σ Rᵢ -0.04043 vs -0.04381 (+7.7 % — larger sum here,
less cancellation-dominated than C2H6's). Wall **226 s vs 192 s = 1.18× slower** — consistent with C2H6's
1.22×.

`‖R‖²` and `max|R|` match to sub-percent with identical nnz — the residual is essentially correct. `Σ Rᵢ`
is 2.1× off, but that is a ~1e-3 near-total-cancellation quantity (the raw R is strongly antisymmetric:
symmetrizing collapses ‖R‖² by ~690×, from 0.118 to 1.7e-4), so a ~0.1 % systematic per-element error
moves it a lot while barely touching ‖R‖². **Not** sparse screening: setting `SPTC_SPARSE_THRESHOLD=0`
leaves the runtime numbers unchanged (the static path segfaults at 0 — dense-intermediate blowup).

**Most likely source of the sub-% gap:** the yielder returns ToT coefficient/amplitude leaves **as-is**
(as MPQC's `eval_csv` does), but the repo's loader stores `c2_tot`/`t2` in pair order `(i₁,i₂)` while the
derivation annotates some occurrences `(i₂,i₁)`. MPQC gets away with as-is because its coefficient storage
already matches the derivation's layout; here the occ pair is transposed on those terms, and the CSV
coefficient is only *nearly* symmetric under i↔j. Fix = permute ToT leaves to `node->annot()` (as the flat
leaves already are), handling the `;`-split outer/inner. Untried; expected to close the gap to ~1e-9.

## The crash that was in the way (root-caused + fixed)

The first working version SIGSEGV'd/heap-corrupted at the μ̃ half-transform. **Root cause: a leaf-yielder
bug, not the evaluator.** The c1/c2 discriminator used `bra_rank()+ket_rank() >= 4`, but CSV coefficients
carry the occupied pair in the PNO index's **proto-indices** (`a<i>` vs `a<i,j>`), so bra+ket = 2 for
*both* — the yielder served `c1_tot` (rank-2 outer, `(9,144)`) for a node annotated with 3 outer indices
`(i₂,i₁,μ̃)`, producing a malformed `Permutation` on the sparse shape → `SparseShape::perm_size_vectors`
segfault. Fixed by discriminating on the PNO proto-index count (c1→1, c2→2).

**How it was localised** (method, for the next such bug): a RelWithDebInfo build of the driver gives full
line info for the header-instantiated `sequant::evaluate` without rebuilding SeQuant; gdb showed the crash
in `SparseShape::perm` (not the earlier release build's inlined `gemm`); an `SPTC_PROD_TRACE` patch in
`result.hpp` then dumped the crashing einsum's operand tranges, exposing the rank-2-array-under-rank-3-
annotation mismatch. A 7-step hand replay (`SPTC_REPRO_EINSUM`) had already proven the contractions
themselves were fine, correctly pointing at the operands the evaluator built.

## Repro

```
cmake -S . -B build-runtime-eval -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=clang++-21 -DCMAKE_C_COMPILER=clang-21 -DSPTC_BUILD_RUNTIME_EVAL=ON
cmake --build build-runtime-eval --target ta_runtime_eval_main -j
D=/proj/perf-model-gpu-PG0/jianjian-scaling/leaves/C2H6_coo   # loader-format (.txt) leaves, NOT .coo.tns
MAD_NUM_THREADS=1 ./build-runtime-eval/ta_runtime_eval_main "$D"       # runs; prints pre-symm + symm checksum + wall
# static reference (symmetrized, to match cck.ipp / the runtime):
MAD_NUM_THREADS=1 SPTC_SYMMETRIZE_R2=1 ./build-runtime-eval/ta_sequant_native_residual_main "$D"
```

Diagnostics: `SPTC_PROD_TRACE=1` (per-einsum annots + tranges), `SPTC_REPRO_EINSUM=1` (7-step hand
replay), `SPTC_NO_CACHE=1`, `SPTC_NO_SYMM=1`, `SPTC_SPARSE_THRESHOLD=<x>`. SeQuant-side diagnostic patch:
`patches/runtime_eval/`.

## Verdict

The reproduction is **built and functionally validated** (crash root-caused and fixed; residual matches to
0.2 %/0.14 %/exact-nnz; a clear, small, known layout item remains to reach bitwise parity). It answers the
question the plan posed — *does reproducing MPQC's runtime evaluator help?* — with a measured **no: 1.22×
slower** than the static generated sequence single-thread, because it runs the same `TA::einsum` with extra
per-node bookkeeping. The value is the faithful control artifact + the measurement.
