# Reproducing MPQC's runtime evaluator (`sequant::evaluate`) — port status + the wall

*2026-08-01. Plan `pure-discovering-book` ("do the reproduce", user chose "Build the port anyway").
Deliverable: `src/ta_runtime_eval_main.cpp` — compute the CSV-CCSD T2 residual by walking the L2
`EvalNode` forest with SeQuant's **runtime evaluator** (`sequant::evaluate`, the way MPQC's `cck.ipp`
does), instead of the repo's STATIC generated einsum sequence (`src/generated_t2_residual.cpp`). Built as
a definitive control artifact; predicted equal-or-worse performance (both funnel through the same
`TA::einsum`).*

## TL;DR / status

The port **compiles, links, derives the equations, and loads all leaves correctly**, but does **not yet
produce a residual**: the runtime evaluator SIGSEGVs / heap-corrupts inside SeQuant's `evaluate`
orchestration at the μ̃-half-transform contraction. The crash is **isolated to SeQuant's evaluator
internals** — the exact contraction sequence, with the evaluator's exact annotations, runs cleanly when
driven by hand (proven below). Closing it needs a debug-symbol SeQuant build + gdb through
`sequant::evaluate`; it is the multi-day-port wall the plan anticipated (risk R1/R4).

## What works (verified on C2H6_coo, single rank, `MAD_NUM_THREADS=1`)

1. **Build/deps.** Unified SeQuant (`SEQUANT_TILEDARRAY=ON`, clang++-21, against the repo's
   `tiledarray-cd53bd3-clang/install`) built + installed to `/users/jianjian/sequant-fork/install-eval`.
   The gated CMake target `ta_runtime_eval_main` (`-DSPTC_BUILD_RUNTIME_EVAL=ON`) resolves SeQuant's
   transitive deps by prepending `install-eval` + the four `build-eval/_deps/{eigen3,range-v3,libperm,
   polymorphic_variant}-build` dirs and `find_package(Eigen3 CONFIG)` **before** `find_package(SeQuant)`.
2. **Derivation reproduces verbatim.** `derive_t2_residual()` mirrors
   `test_csv_ccsd_derivation.cpp:265-531` (`make_sr_spaces` → `make_cceqvec_csv_closedshell(2)` →
   `tail_factor` → `density_fit(…,Κ)` → `csv_transform(…,μ̃,"C")` → `flatten` → `optimize` with
   `OptFor::Flops`, `ReorderSum::Reorder`, volatile-leaf=t, n_replay=10). Yields **55 summands**.
3. **Forest + cache build.** Per-summand `binarize<EvalExprTA>(ResultExpr{make_R_template_csv(2),
   summand})`; `cache_manager(nodes, is_t_leaf, 2)`.
4. **Leaf yielder maps every leaf.** `t`/`C` returned as ToT (as MPQC's `eval_csv` does — verified it
   returns amplitudes/coeffs as-is, no permute); flat `g`/`f`/`s` permuted `res(node->annot()) =
   field(temp_annot)`. **No unmapped-leaf throw.** All leaves load with correct nnz — `t_i_i_a_a`
   **nnz=150301**, matching the static path exactly.

## The crash — precisely isolated

The evaluator dies producing the μ̃ half-transform. With per-contraction annotation tracing patched into
SeQuant's `result.hpp` (`SPTC_PROD_TRACE`), summand 0 reaches:

```
[prod-ToT] L='i_3,μ̃_19579;a_3i_3'  R='i_3;a_3i_3'          -> 'μ̃_19579,i_3'            (C1×t1, DeNest)
[prod-flat] L='i_3,μ̃_19579,Κ_1'    R='μ̃_19579,i_3'         -> 'Κ_1'                     (×g)
[prod-flat] L='Κ_1'                 R='μ̃_19580,μ̃_19581,Κ_1' -> 'μ̃_19580,μ̃_19581'        (×g0)
[prod-ToT] L='i_2,i_1,μ̃_19580;a_2i_1i_2' R='μ̃_19580,μ̃_19581' -> 'i_2,i_1,μ̃_19581;a_2i_1i_2'  <-- CRASH
```

The crash is the **flat(μ̃,μ̃) × ToT-C2 → ToT** step (`SparseShape<float>::gemm`, non-deterministic
SIGSEGV / `free(): corrupted unsorted chunks` / integer-divide-by-zero — classic dangling/uninitialised
memory). The **static** path forms this exact contraction and works
(`generated_t2_residual.cpp:84-85`: build `I_μ̃_μ̃` from g0, then `× C_ap2_μ̃`).

### Proof the math/data/TA are NOT at fault

`ta_runtime_eval_main` embeds a reproducer (`SPTC_REPRO_EINSUM=1`) that replays summand 0's exact
7-step chain (s1 `C1×t1`-DeNest → s2 `×g` → s3 `×g0` → s4 `×C2` → s5 double-inner `×C2` → s6 `×t2`)
against the **real loaded leaves**, with the evaluator's **exact annotations** (incl. the swapped occ
order `i_2,i_1` and pair-subscripted inner `a_2i_1i_2`). **Every step returns OK**, under every
variation tried:

| variation | result |
|---|---|
| clean labels / evaluator's exact labels | all OK |
| flat-first / ToT-first operand order | all OK |
| occ order `i,j` / swapped `j,i` | all OK |
| with / without inter-step `world.gop.fence()` | all OK |
| `SPTC_NO_CACHE` (CacheManager disabled) | still crashes *in the evaluator* |
| `SPTC_CLONE_LEAF` (deep-clone ToT leaves, no aliasing) | still crashes *in the evaluator* |

Also ruled out by measurement: **tiling** — g0's μ̃ and c2_tot's μ̃ are identical TiledRanges (extent
144, 8 tiles of 18; `adaptive_tile_sizes` keys tile size to extent alone, so every μ̃ axis matches).

**Conclusion:** the contraction, its data, its tiling, and `TA::einsum` are all correct. The fault is
inside SeQuant's `evaluate` recursion / `Result` object lifetimes — an intermediate array the evaluator
builds is subtly malformed or freed-too-early relative to the hand-driven equivalent. Black-box testing
cannot localise it further; the next step is a `-g` SeQuant build + gdb inside `sequant::evaluate` (or
valgrind, to catch the first invalid write, which the non-determinism says precedes the detected crash).

## One real sub-finding (not the root cause)

SeQuant's TA backend emits the mixed `ToT * T -> ToT` contraction as `einsum(ToT, flat)` —
**ToT operand first** (`result.hpp:610`). This TiledArray fork's de-nesting einsum is **not
operand-order-symmetric** there: `einsum(ToT, flat)` SIGSEGVs immediately in `SparseShape::gemm`, while
`einsum(flat, ToT)` (the order the static generator always emits, e.g. `generated_t2_residual.cpp:85`)
gets further before heap-corrupting. Patched `result.hpp` to emit flat-first (value-identical; operand
listing is commutative). This only **changes the failure mode**, it does not fix the evaluator crash —
kept because flat-first matches the proven static convention.

## Repro / next session

```
cmake -S . -B build-runtime-eval -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=clang++-21 -DCMAKE_C_COMPILER=clang-21 -DSPTC_BUILD_RUNTIME_EVAL=ON
cmake --build build-runtime-eval --target ta_runtime_eval_main -j
D=/proj/perf-model-gpu-PG0/jianjian-scaling/leaves/C2H6_coo   # loader-format (.txt) leaves
MAD_NUM_THREADS=1 SPTC_WARMUP=0 ./build-runtime-eval/ta_runtime_eval_main "$D"          # crashes at μ̃-transform
MAD_NUM_THREADS=1 SPTC_REPRO_EINSUM=1 ./build-runtime-eval/ta_runtime_eval_main "$D"    # 7-step replay: all OK
MAD_NUM_THREADS=1 SPTC_NO_CACHE=1 SPTC_PROD_TRACE=1 ./build-runtime-eval/ta_runtime_eval_main "$D"  # annot trace
```

SeQuant-side diagnostic patches (operand-swap + `SPTC_PROD_TRACE`) are saved under
`patches/runtime_eval/`. Static reference to match once it runs (C2H6_coo, 1-thread):
`nnz=150301 sum=-0.000677185440986902 sumsq=0.117959336446712 max_abs=0.0170233601671088`, wall≈42.7 s.

## Verdict

The reproduction is ~80% built (derivation + forest + leaf mapping + build/deps all done and verified)
and stopped at a well-characterised bug **inside SeQuant's evaluator**, not in the repro's data or TA.
This matches the plan's honest prediction (equal-or-worse, a multi-day port); the value delivered is the
faithful port scaffold + the precise isolation, so a debug-symbol SeQuant session continues from the
exact failing frame rather than from scratch.
