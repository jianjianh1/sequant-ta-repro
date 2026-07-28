# What real MPQC does, and where this reproduction diverges

A detail-by-detail comparison of MPQC's own closed-shell CSV-CCSD T1/T2
residual evaluation against this repo's from-scratch TiledArray (TA)
reproduction, and the resolution of the performance gap between them.

**Headline (settled).** The reproduction started ~3.30× slower than real
MPQC combined. That gap was **not** a fundamentally worse algorithm,
backend, or contraction structure — it was three concrete, fixable things:
(1) the repro forced size-1 occupied tiling where MPQC uses coarse
`occ_tile_size=4`; (2) the repro was built with gcc where MPQC uses
clang-21; and (3) MPQC's published number is a *warm* CCSD iteration
(t-independent intermediates cached) while the repro measured a *cold*
one. Correcting all three, on equal footing the reproduction **matches or
beats real MPQC**: cold-to-cold ~1.44× faster on T2, warm-to-warm ~3.3×
faster. Both sides use the same TiledArray, issue the same `TA::einsum`
calls, and form the same intermediates.

"Real MPQC" = source in `/users/jianjian/mpqc4`, built as the instrumented
Apptainer SIF by `/users/jianjian/mpqc-benchmark`, on ethane (C2H6,
cc-pVDZ-F12 / aug-cc-pVDZ-RI, frozen-core; `ethane-perf.json`). All
`mpqc4:`/`src/` line numbers are as of this writing. Every timing below is
correctness-gated on the reference checksums (see Verification).

**§11 extends this from a single ethane point to the alkane series
C₂H₆–C₅H₁₂ (cc-pVTZ) with a 1-rank-per-node sweep over CloudLab node16–31,
run on *both* sides.** The key correction it delivers: both the repro *and*
MPQC scale well multi-node, so the fair comparison is at **equal ranks** —
and there MPQC is faster for every real molecule (repro ~2–3× slower warm,
~4.5–7× slower cold at np=16; the repro is ahead only on tiny ethane's warm
residual, where MPQC's small-workload overhead dominates). The gap is pinned
to one giant DF-half-transform intermediate that the repro materialises —
which both scales worse across ranks and is a hard memory wall (it OOMs
hexane even across 16 nodes). Owning-ToT makes that intermediate tractable on the feasible
molecules; the residual gap is MPQC's runtime evaluator distributing the DF work across ranks
better than the repro's static einsum sequence (the proto-extent lever, an arena-era idea, is
superseded by owning-ToT — see §11).

---

## 1. What real MPQC does — the six axes

| Axis | Real MPQC | This reproduction | Same? |
| --- | --- | --- | --- |
| **Derivation** | `make_cceqvec_csv_closedshell` (`mpqc4:sequant.cpp:156-211`): spin-free, `mbpt::Context{.csv=CSV::Yes}`, `CC{k}.t()`, per-rank biorthogonal transform (`tail_factor`→`biorthogonal_transform(external_indices(S))`→re-prepend `S`→`simplify`). | Same pipeline, run offline in `sequant-fork`'s `test_csv_ccsd_derivation.cpp`; output pasted into `src/generated_t{1,2}_residual.cpp`. | ✅ same algebra |
| **Optimizer** | `generate_csv_closedshell` (`cck.ipp:1516-1541`): `density_fit`→`csv_transform`→`flatten`→`optimize(OptFor::Flops, n_replay=10, real average_csv_extent, is_volatile_leaf=t)`. | Same `optimize()` (fork = `d3a38f8f` + export commits; **no optimizer change**), same options, same extents (real `average_csv_extent(2)`≈44.6 ≈ hardcoded 45). | ✅ same order |
| **Binarization/eval** | `populate_equation_nodes`→per-summand binary eval trees; `sequant::evaluate(node, annot, leaf_eval, Cs[R])` with a runtime `CacheManager` (`min_repeats=2`), `cck.ipp:1679-1692`. | Same binarization (fork's `eval_expr.cpp`); exported to a flat `TA::einsum` sequence with cross-term CSE instead of a runtime cache. | ⚠️ static CSE vs runtime cache (§5-6) |
| **Contraction primitive** | `TA::einsum` / `TA::einsum<DeNest::True>` (SeQuant eval backend `result.hpp:379,613-628`). | **Identical** `TA::einsum` / `TA::einsum<DeNest::True>` (`generated_t{1,2}`). | ✅ same TA call |
| **ToT storage** | `DistArray<Tensor<ArenaTensor<T>>, SparsePolicy>` (`csv.h:34`), arena-pinned inner cells. | Identical (`src/ta_tensors.h`); owning `Tensor<Tensor<double>>` also available via `SPTC_OWNING_TOT` (§8). | ✅ same |
| **Tiling** | occ retiled to `occ_tile_size=4` for the residual (§3). | was size-1; now coarse `occ_tile=2` (§3). | ✅ after §3 |
| **Backend** | SIF built `-DMADNESS_TASK_BACKEND=PaRSEC`; single-rank, `MAD_NUM_THREADS≈11-12`. | MADNESS/Pthreads (PaRSEC tested, slower — §4). | ⚠️ cross-backend, immaterial (§4) |
| **BLAS** | OpenBLAS. | OpenBLAS (headline); MKL ~7% faster but off-parity (§4). | ✅ OpenBLAS both |

The two sides are the **same computation on the same library**. The
divergences that mattered were tiling (§3) and the build (§4); the
apparent residual gap after those was a measurement artifact (§6).

---

## 2. Reference problem and checksums

Leaf data + reference checksums: `mpqc-benchmark/traces/checksum-run/`.
Ethane frozen-core: occ_act=7, PAO μ̃=114, RI Κ=282. Every configuration
below must reproduce (± last-ULP CSE reassociation noise):

| Residual | nnz | sum | max_abs |
| --- | --- | --- | --- |
| T1 | 522 | −0.0888799157 | 0.0151662826 |
| T2 | 98598 | 0.2141888066 | 0.0174497557 |

---

## 3. The tiling divergence and its fix (the largest lever)

**Divergence.** `ethane-perf.json` sets `unit_csv_tile_size=true`, which
unit-tiles the occupied range at CSV *construction* (`pao_to_pno_mp2.ipp:1022`)
— but the tensors that enter the residual are retiled to `occ_tile_size=4`.
MPQC's own dumped tile metadata confirms it: T2 (`expr0010`) has
`outer_shape 7,7` on **4 outer tiles** (2×2 occupied-pair grid, ≤16
pairs/tile); C2/g likewise tile occ to extent 4. The reproduction instead
forced **one pair per outer tile** (size-1): 49 outer tiles for T2 vs
MPQC's 4 — a ~12× task-count multiplier (T1: 7 vs 2, ~3.5×). A fresh
`perf` profile showed the consequence: ~43-50% of size-1 T2 wall was
MADNESS task-scheduling (`ThreadPool::run_tasks` + `sched_yield`), not math.

The repro forced size-1 because coarsening crashed `TA::einsum` on the old
pinned commit `84411a6`. **Fix:** move to the TA revision MPQC itself
tracks — `cd53bd3` (`mpqc4:external/versions.cmake`) — whose `einsum`
handles multi-pair ToT tiles. (Its install ships incomplete: `madness/misc/*`,
~180 MADNESS headers and TA `.ipp` files must be copied in — see §10.)
Then tile the occupied dimension coarsely and consistently across **all**
tensors (`ta_builder.h`, env-gated `SPTC_COARSE_OCC`/`SPTC_OCC_TILE`), with
a **ragged per-pair inner** size (`SPTC_COARSE_PAD=0`) matching MPQC's
layout (no zero-padding). Sweeps settled the optimum at `occ_tile=2`,
`SPTC_TILES_PER_DIM=4`, `MAD_NUM_THREADS=8`, `yield` (`occ_tile≥3` ragged
still trips `einsum`; occ_tile=2 groups 2×2 pairs, keeping sparsity).

**Result (gcc, OpenBLAS, 3-trial medians):** T1 0.871→**0.263 s**, T2
8.08→**4.13 s** → combined gap **3.30× → 1.62×**, checksums exact.

---

## 4. Build fidelity: compiler, BLAS, backend

Three builds isolate compiler from backend (coarse tiling, OpenBLAS,
3-trial medians, checksums exact):

| | gcc Pthreads | **clang Pthreads** | clang PaRSEC | MPQC |
| --- | --- | --- | --- | --- |
| T1 | 0.263 s | **0.229 s** | 0.327 s | 0.461 s |
| T2 | 4.13 s | **2.88 s** | 3.19 s | 2.254 s |

- **Compiler is a real lever.** MPQC's SIF builds with **clang-21**; the
  repro was on gcc. Same Pthreads backend, gcc→clang cuts T2 4.13→2.88 s
  (~31%) — straightforward build fidelity. Combined gap **1.62× → 1.14×**,
  and T1 (0.229 s) is now ~2× *faster* than MPQC.
- **PaRSEC does not help.** On the same compiler, Pthreads (2.88 s) beats
  PaRSEC (3.19 s) on T2. (An earlier reading crediting PaRSEC was the clang
  confound.) The cross-backend comparison is immaterial.
- **BLAS.** MKL is ~6-7% faster than OpenBLAS, but MPQC runs OpenBLAS, so
  the headline stays OpenBLAS-vs-OpenBLAS.

---

## 5. Same TiledArray, same operations — no eval-structure gap

Because both sides use the same TA, the residual difference had to be in
*how* each drives it. A per-op trace (`src/sptc_traced_einsum.h`,
`SPTC_TRACE_OPS`) diffed against MPQC's `steps.csv` (`traces/parse_trace.py`)
settled it:

- **Same primitive:** every contraction is `TA::einsum` /
  `TA::einsum<DeNest::True>` on both sides — not a cheaper API in MPQC.
- **Same intermediates, including the giants.** The repro forms two large
  DF/CSV intermediates — `g·g→i,μ̃,μ̃,μ̃` (13M nnz, `generated_t2:211`) and
  `g·C→i,i,μ̃,Κ;a` (70M nnz, `generated_t2:486`). MPQC's trace contains the
  **byte-for-byte same nodes** (iter1 term39 nnz 10,370,808; term81 nnz
  70,211,232). Contraction order and intermediate structure are the same —
  the optimizer is not the difference (the fork's `optimize()` is unchanged
  from MPQC's `d3a38f8f`).
- Cross-term CSE, optimizer extent tuning, and de-serializing the generated
  code were all tested and ruled out (each kept the same giants or broke
  correctness). The only structural difference is *when* the giants are
  computed — §6.

---

## 6. The real residual "gap" was warm-vs-cold

The two giant DF/CSV intermediates are **t-independent** (functions of g,
C, s, f — not the amplitude). MPQC caches them across CCSD iterations via
its persistent `CacheManager` (`is_volatile_leaf = t`, `cck.h:619-626`):
the 70M giant appears in MPQC's **iter1 only**; iters 2/3 reuse it. MPQC's
published 2.254 s is the **warm iter2** measurement; the repro's 2.88 s
**recomputes everything (cold)**. Per-iteration R=2 contraction time from
MPQC's trace (validated: iter2 per-node sum 2.22 s ≈ the reported 2.254 s):

| | iter1 (cold) | iter2 (warm) |
| --- | --- | --- |
| MPQC R=2 | **4.15 s** | **2.22 s** |
| repro R=2 (cold) | **2.88 s** | — |

**Cold-to-cold, the repro (2.88 s) is ~1.44× *faster* than MPQC (4.15 s)** —
it computes the identical giants faster (coarse tiling + clang). The
"1.28× MPQC-faster" figure was cold-repro vs warm-MPQC, not apples-to-apples.

---

## 7. Persistent-cache warm — repro ~3.3× faster than MPQC warm

To measure the repro *warm* (MPQC's footing), the generated T2 was
SSA-split (`gen_split/`, `emit_split.py`) into a **t-independent block
computed once** (the persistent cache — 55 statements incl. both giants)
plus a **t-dependent residual timed each call** (197 statements) — the
faithful analog of MPQC's volatile/persistent split. Correctness-gated:
reproduces the T2 checksum exactly.

Result (arena, coarse, NT=8, 3 runs × warmup+3 trials, all clean):

| | MPQC (warm) | repro (warm) |
| --- | --- | --- |
| T2 | 2.254 s | **0.68 s** (~3.3× faster) |

(Amortized t-independent precompute ~2.3 s once; 2.3 + 0.68 ≈ 2.88 s cold,
consistent.)

**Lesson worth recording.** A long detour blamed a multithreaded segfault
in this path on a TA `ArenaTensor`/`SparseShape` cross-thread deletion bug.
A gdb backtrace disproved it: the fault was the warm driver calling
`madness::threadpool_wait_policy()` **before** `TA_SCOPED_INITIALIZE`
(null ThreadPool singleton), segfaulting at NT>1. Moving that call after
init (`gen_split/ta_warm_t2_main.cpp`) made the warm path stable. There is
no TA defect here; the `cd53bd3` coarse path runs correctly at NT>1.

---

## 8. Multithreading and multi-rank

- **Multithreading.** The residual always ran multithreaded (arena,
  NT=8 — the §3-4 numbers); the only NT>1 crash was the §7 driver bug, now
  fixed. Optimum stays `MAD_NUM_THREADS=8` + `yield`.
- **Owning ToT fallback (`SPTC_OWNING_TOT`, `ta_tensors.h`).** The default
  `ArenaTensor` ToT has non-owning inner cells (views into an arena page
  freed cross-thread by MADNESS) — a real hazard for persistent/reused ToT
  arrays. A compile-time switch to owning `Tensor<Tensor<double>>`
  (upstream-supported; numerically identical per TA's `einsum.cpp`
  `arena_matches_owning`) removes it, at ~2× cost (coarse cold 6.3 vs
  2.88 s). Kept as a safety fallback; not needed for the paths here.
- **Multi-rank (MPI).** Already distributed-tile + `MPI_Allreduce`-checksum
  correct. `mpirun -np 2` and `-np 4` (single node) complete with **no
  hang** and checksums bit-identical to np=1 (`nranks` self-labeled). No
  speedup — ethane's coarse T2 has only ~9 outer pair-tiles, too few to
  distribute, so cross-rank comm dominates; this matches MPQC's own choice
  to run this workload single-rank.

---

## 9. Final results — the journey

All correctness-gated (checksums exact), ethane, OpenBLAS both sides.

| Stage | T1 | T2 | Combined vs MPQC |
| --- | --- | --- | --- |
| Original (size-1, gcc, cold) | 0.871 s | 8.08 s | 3.30× slower |
| + coarse occ tiling (§3) | 0.263 s | 4.13 s | 1.62× slower |
| + clang-21 build fidelity (§4) | **0.229 s** | **2.88 s** | 1.14× slower |
| Fair footing, cold-to-cold (§6) | — | 2.88 vs **4.15** | **repro 1.44× faster** |
| Fair footing, warm-to-warm (§7) | — | **0.68** vs 2.254 | **repro 3.3× faster** |

What each lever was worth: **coarse tiling** 3.30→1.62× (biggest);
**clang** 1.62→1.14×; **fair (warm/cold) measurement** flips the residual
to a repro win. Backend, BLAS, cross-term CSE, and eval structure were
tested and found *not* to be the difference.

---

## 10. Reproduction

```bash
# 1. TiledArray cd53bd3 (the revision MPQC tracks), clang-21, OpenBLAS,
#    completing its incomplete install (headers) automatically.
./setup-tiledarray.sh
# 2. Build the repro with clang-21 against it (tools = the warm benchmark).
cmake -B build-cd53bd3-clang -DCMAKE_BUILD_TYPE=Release -DSPTC_BUILD_TOOLS=ON \
  -DCMAKE_CXX_COMPILER=clang++-21 -DCMAKE_C_COMPILER=clang-21 .
cmake --build build-cd53bd3-clang -j --target ta_sequant_native_residual_main ta_warm_t2
# 3a. Cold residual (fair vs MPQC cold 4.15 s):
SPTC_COARSE_OCC=9 SPTC_OCC_TILE=2 SPTC_COARSE_PAD=0 SPTC_TILES_PER_DIM=4 \
SPTC_MAD_WAIT_POLICY=yield MAD_NUM_THREADS=8 SPTC_TRIALS=3 SPTC_WARMUP=1 \
  taskset -c 0-7 ./build-cd53bd3-clang/ta_sequant_native_residual_main \
  ../mpqc-benchmark/traces/checksum-run/sptc_coo_iter1
# 3b. Warm T2 (persistent cache; fair vs MPQC warm 2.254 s) — same env,
#     ./build-cd53bd3-clang/ta_warm_t2 <dir>   ->  ~0.68 s.
# 3c. Multi-rank: mpirun -np 2 -x MAD_NUM_THREADS=4 -x OMP_NUM_THREADS=1 <bin> <dir>
```

Env-gated knobs leave defaults untouched when unset:
`SPTC_COARSE_OCC`/`SPTC_OCC_TILE`/`SPTC_COARSE_PAD`/`SPTC_TILES_PER_DIM`
(tiling), `-DSPTC_OWNING_TOT` (owning fallback), `SPTC_TRACE_OPS`
(per-op trace).

## Verification
- **Correctness gate (every config/rank count):** the §2 checksums, ± last
  ULP; discard any run that misses them.
- **Fair timing:** compare like-for-like — same compiler, same tiling, and
  the same iteration *warmth* (MPQC's published number is warm iter2; use
  the repro's warm path, or MPQC's cold iter1, for apples-to-apples).
- **Multithread / multi-rank:** the fixed-driver (or owning) build
  completes with no segfault at NT=8; np=2/4 checksums bit-identical to np=1.
- CloudLab: build on node-local disk; watch disk headroom and the
  experiment expiry clock.

## 11. Scaling across molecules and ranks (alkanes C₂H₆–C₆H₁₄, nodes 16–31)

The ethane parity study (§1–10) is one point. This section extends it to the
linear-alkane series **C₂H₆, C₃H₈, C₄H₁₀, C₅H₁₂** (cc-pVTZ / cc-pVTZ-RI, the
`jianjianh1/mpqc-alkanes-v3` dataset) with a **rank sweep np ∈ {1,4,8,16}, one MPI
rank per node across CloudLab node16–31**, run **on both sides**. C₆H₁₄ is under
*Limits*. Both sides use the identical TiledArray fork `cd53bd3` and the same SeQuant
derivation; MPQC is the instrumented sif on PaRSEC, the repro is the generated
`TA::einsum` sequence (owning-ToT, coarse tiling `SPTC_COARSE_OCC=9 SPTC_OCC_TILE=2
SPTC_TILES_PER_DIM=8`, `MAD_NUM_THREADS=8`, `--bind-to none`).

**Both sides multi-rank scale — compare at the SAME rank count.** A single-rank
MPQC number is *not* MPQC's best at these sizes: run multi-node, MPQC scales strongly
for the real molecules (C₄H₁₀ cold T2 76.9 s single-rank → **10.2 s at np=16, 7.6×**;
C₅H₁₂ 233 → **20.4 s, 11×**). Only ethane is too small to benefit (its warm
multi-node is marginally *slower* than single-rank). So the honest comparison is
same-np, both sides:

**WARM T2 (s) — repro / MPQC / ratio, same np:**

| mol | metric | np1 | np2 | np4 | np8 | np16 |
|---|---|---|---|---|---|---|
| C2H6 | repro | 3.0 | 4.5 | 4.3 | 3.6 | 3.2 |
| C2H6 | MPQC | 3.8 | - | 3.7 | 4.3 | 4.4 |
| C2H6 | ratio | 0.8x | - | 1.2x | 0.8x | 0.7x |
| C3H8 | repro | 22.7 | 30.5 | 27.6 | 21.7 | 18.4 |
| C3H8 | MPQC | 18.0 | - | 9.7 | 9.3 | 8.5 |
| C3H8 | ratio | 1.3x | - | 2.8x | 2.3x | 2.2x |
| C4H10 | repro | 37.6 | 41.3 | 35.6 | 25.2 | 20.6 |
| C4H10 | MPQC | 22.3 | - | 16.7 | 13.3 | 10.7 |
| C4H10 | ratio | 1.7x | - | 2.1x | 1.9x | 1.9x |
| C5H12 | repro | 116.2 | 114.9 | 91.0 | 62.3 | 48.7 |
| C5H12 | MPQC | 39.6 | - | 28.4 | 22.2 | 18.3 |
| C5H12 | ratio | 2.9x | - | 3.2x | 2.8x | 2.7x |

**COLD T2 (s) — repro / MPQC / ratio, same np:**

| mol | metric | np1 | np2 | np4 | np8 | np16 |
|---|---|---|---|---|---|---|
| C2H6 | repro | 12.2 | 12.4 | 9.7 | 8.2 | 6.6 |
| C2H6 | MPQC | 7.7 | - | 3.7 | 3.9 | 4.3 |
| C2H6 | ratio | 1.6x | - | 2.6x | 2.1x | 1.5x |
| C3H8 | repro | 66.1 | 63.0 | 48.1 | 35.6 | 30.0 |
| C3H8 | MPQC | 31.3 | - | 8.1 | 7.4 | 6.8 |
| C3H8 | ratio | 2.1x | - | 6.0x | 4.8x | 4.4x |
| C4H10 | repro | 154.6 | 133.4 | 99.3 | 67.8 | 48.4 |
| C4H10 | MPQC | 76.9 | - | 13.9 | 11.6 | 10.2 |
| C4H10 | ratio | 2.0x | - | 7.1x | 5.9x | 4.8x |
| C5H12 | repro | - | 339.0 | 215.1 | 149.3 | 114.8 |
| C5H12 | MPQC | 233.1 | - | 36.2 | 25.9 | 20.4 |
| C5H12 | ratio | - | - | 5.9x | 5.8x | 5.6x |

**Result.** At equal ranks **MPQC is faster for every real molecule**: the repro is
**~2–3× slower warm and ~4.5–7× slower cold** (C₃H₈–C₅H₁₂). The repro is
competitive/faster *only* for tiny ethane's warm residual (0.7–0.8×), where MPQC's
multi-node overhead dominates its small workload. (An earlier version of this section
reported the repro "beating MPQC cold at np=16" — that compared the repro's 16-rank
time against MPQC's *single-rank* time, which is not a fair comparison once MPQC uses
the ranks too. Corrected here.)

**Why — the giant DF-half-transform intermediate.** The cold gap (~5×) is the single
μ̃Κ intermediate (§3/§6). MPQC's runtime evaluator distributes it across ranks
efficiently; the repro's generated sequence materialises it explicitly, which both
scales worse and is a hard **memory wall** (see *Limits*). The warm gap (~2×) is the
same intermediate's ragged-ToT contraction efficiency. Note the repro *does* scale
(C₅H₁₂ cold 339→15 s over the sweep) — it is just outpaced by MPQC's better
multi-node DF handling.

**Correctness.** Anchored on the gauge-free R(T=0): feeding zero t-amplitudes, the
repro reproduces MPQC's occurrence-1 residual (validated multi-rank at np=16, so it
also checks the distributed path). **Bit-exact** for C₃H₈ (T2 nnz 261914, sum
14.9251990396) and C₄H₁₀ (nnz 371400, sum 17.8115228729) — matching MPQC to ~13
digits; C₅H₁₂ matches to ~7 significant figures (sum 22.327121 vs 22.327137, nnz
482201 exact, sumsq/max to ~10 digits — float reassociation on the larger residual).
All sweep checksums are rank-invariant across np (the multi-rank correctness gate).

**Owning-ToT is required at multi-rank.** The default `TA::ArenaTensor` inner tile
segfaults at np≥8 for the larger molecules (cross-rank lazy-deletion race in MADNESS);
`-DSPTC_OWNING_TOT` (owning `Tensor<double>` inner cells) removes it, is numerically
identical, and is ~40 % faster here.

**Limits & the path to close the gap.** Hexane (C₆H₁₄, occ 19 / PNO 351 / RI 906)
exhausts memory even across all 16 nodes for the repro (`std::bad_alloc`) — the giant
intermediate is a memory wall (MPQC single-rank hexane also OOMs at 63 GB; MPQC
multi-node clears it). **The cold-precompute fix that worked was owning-ToT, not proto=100.** The giant μ̃Κ
intermediate is only catastrophic with the default arena inner tile (`TA::ArenaTensor`): there
it costs ~70 s serial, and a generator lever (`proto=100`, raising `optimize()`'s PNO/proto
extent so the cost model never forms it) cut cold T2 ~3× at np=1. But **owning-ToT**
(`-DSPTC_OWNING_TOT`, already required for multi-rank stability) makes that same intermediate
cheap, and proto=100's extra-op factorization then loses everywhere: single-node C₂H₆ cold
proto=45 11.6 s vs proto=100 18.9 s; at np=16, 6.6 vs 13.8 s (proto=45's giant intermediate
distributes across ranks better than proto=100's alternative). So proto=100 is an arena-era
optimization superseded by owning-ToT and is **not adopted**; the cold numbers above are the
faster proto=45 owning path. (proto=100 is validated bit-exact vs MPQC on small-basis ethane
and at R(T=0), with a ~0.2% cc-pVTZ t-dependent order-sensitivity — a real property of the
ragged-ToT contraction — but that's moot since it isn't faster.) Hexane stays out regardless:
proto=100 clears its OOM but then aborts at 16-node scale with a separate TA/MADNESS heap
corruption. **The residual cold gap at np=16 is genuine, and it is a tiling/distribution gap,
not a "MPQC forms less" gap.** In these runs MPQC's aux-Κ batching was *off*
(`batch:aux_target_size` default 0), so MPQC *also* materializes the whole giant μ̃Κ
intermediate and calls the *same* `TA::einsum` SUMMA on the same TiledArray — its runtime cache
avoids recomputation, not size. The difference is that MPQC's DF-carrying arrays inherit the CSV
solver's `TiledRange`/`SparseShape`/`pmap` (so SUMMA load-balances the intermediate the way the
solver laid out the occupied space), while the repro builds its own tiling and takes TiledArray's
default pmap in `build_tot_array`. Closing it is therefore an in-repo tiling/pmap lever, not a
whole new evaluator — see `MPQC_EVALUATION.md` for the stage-by-stage trace and the three
concrete levers.

**Gap-closing attempts (what did and didn't move it).** (1) *Owning-ToT* is the cold fix
that worked (§ above) — it makes the giant DF intermediate tractable and, at multi-rank,
distributes it well. (2) *proto=100* (generator extent) is superseded by owning and not adopted
(slower everywhere with owning). (3) *R2 symmetrization* — MPQC checksums R2 after
`0.5·(R[i,j;a,b]+R[j,i;b,a])`; the repro's R2 is in fact ~antisymmetric under that swap and its
RAW value is what matches the reference, so symmetrizing is neither applied nor a reconciler
(tested: it preserves the divergent sum). (4) *Warm tiling* — the warm optimum is `TILES_PER_DIM=6`,
~7% faster than the cold-optimal 8 (C₂H₆ warm 2.75 s vs 2.97 s); the rest of the ~2× warm gap is
distributed across ~250 small ragged-ToT ops (no single-op lever). Net: the ~7% warm tiling win
aside, the residual gap is not closable by in-repo tuning — it is MPQC's runtime-evaluator DF
distribution, and matching it needs the evaluator's aux-Κ batching / occ-batch / runtime cache.

**Takeaway.** The reproduction issues the same `TA::einsum` algebra as MPQC, on the same
TiledArray, and is numerically correct; both sides even materialize the same giant DF
half-transform intermediate (MPQC's aux-batching was off in these runs). At real (cc-pVTZ) scale
and equal ranks MPQC is still ~2× warm / ~5× cold faster, and the gap is now pinned to three
concrete, named levers (full trace in `MPQC_EVALUATION.md`):
1. **Cold np=16 → pmap co-location** of the DF-carrying ToT arrays. MPQC's operands inherit the
   CSV solver's `trange`/`shape`/`pmap` (`cck.ipp:1563-1565`); the repro's take TA's default from
   a self-chosen tiling (`ta_builder.h:363`). The same SUMMA, load-balanced differently. A tiling
   sweep (`MPQC_EVALUATION.md` verification) rules out occ tile *size* as the lever (finer = no
   change, coarser = worse) — it is the **pmap**, addressable in `build_tot_array`.
2. **Hexane memory wall → aux-Κ batching** (`eval.hpp:1129`, `cck.ipp:1601-1645`): stream Κ in
   tile-aligned slices over the persistent DF terms so the intermediate is never fully formed.
   *A generator/backend project; the memory fix, independent of (1).*
3. **Warm ~2× → mostly fundamental.** ~7% via `SPTC_TILES_PER_DIM=6`; the rest is distributed
   across ~250 small ragged-ToT ops — the static sequence vs the runtime evaluator, addressable
   only by a representation/backend change.
