# Fair, practical comparison with MPQC

This is the fair-comparison methodology + results (plan: a two-level comparison — per-operation isolated
einsum suite as primary, whole-residual as secondary). It supersedes the confounded historical numbers in
`docs/MPQC_COMPARISON.md` for the single-node cold cache-free question.

## Fairness principles (why these numbers are honest)

- **Same TiledArray.** Both sides link the *identical* TA commit `cd53bd3` (same `jianjianh1/tiledarray`
  fork), same clang-21, same Release, same OpenBLAS, same OpenMPI. The only build divergence is the MADNESS
  task backend — MPQC=PaRSEC, repo=Pthreads — measured at ~1% single-node (`docs/MPQC_ABLATION.md:83-89`).
- **Cache-free both sides.** Repo runs the `SPTC_NO_CSE` default sequence (no cross-term reuse); MPQC runs
  `cache_imeds=false` (no intermediate memoization). `seq_opt` stays ON on both (the shared SeQuant
  factorization; off ⇒ OOM).
- **Matched threads + hygiene.** Both at **8 threads** (MPQC: "7 worker + 1 main"; repo: `MAD_NUM_THREADS=8`,
  `taskset -c 0-7`), BLAS pinned to 1 thread (`OMP_NUM_THREADS=1` / `OPENBLAS_NUM_THREADS=1`), performance
  governor, turbo off, page cache dropped. Same molecule/basis/leaves (ethane, cc-pVDZ-F12 / aug-cc-pVDZ-RI,
  frozen-core), single rank.
- **Correctness-gated.** Repo residual reproduces the reference checksum exactly (T2 nnz=98598,
  sum=0.2141888066, sumsq=0.0997316723, max_abs=0.0174497557). (Signed `sum` is not a fair gate on ethane —
  degenerate orbitals give a gauge sign ambiguity; gate on nnz/sumsq/max_abs.)

## Level 2 (secondary): whole-residual, cold, cache-free — RESULT

MPQC's trace-independent `Eval | WholeResidualWallTime | R=2` (`cck.ipp:1772-1779`, built *"for a fair
performance comparison against a native reproduction"*) vs the repo's fenced `whole_t2_residual wall_s` —
the same construction (wall through the terminating fence).

| Whole **T2** residual (ethane, cold, cache-free, 8 threads, np=1) | wall |
| --- | --- |
| Repo (cache-free stock owning TA), median of 5 trials | **14.02 s** |
| MPQC (`cache_imeds=false`, cold iter1) | **11.11 s** |
| **Ratio (repo / MPQC)** | **1.26×** (MPQC faster) |

T1 (cold cache-free): MPQC 1.21 s. Both checksums match.

**Reading it honestly.** The repo is ~1.26× slower on the cold cache-free whole T2 residual at 8 threads,
single node — within/near MPQC's own run-to-run variance (~±20-30% on warm T2; single cold sample here).
This is *not* a per-op or memory-bandwidth gap (the deep-profile docs measure both sides compute-bound,
issuing the same ~170k tiny per-pair GEMMs); it is MPQC's runtime evaluator keeping the threadpool BLAS-fed
(≈44% dgemm self-time) vs the repo's finer-grained ToT tasks (thread-starvation at 8 threads). Cache-free
"later iterations" cost *more* on the MPQC side too (iter2 T2 = 16.46 s > iter1 11.11 s), the expected
no-reuse behavior — so the repo's every-trial-cold 14.02 s is the right analog to MPQC's cold iter1.

**Retired / superseded numbers.** The historical "repro 1.44× / 3.3× faster" (`MPQC_COMPARISON.md` §6-7,
ethane cc-pVDZ-F12) mixed cold-repro vs warm-MPQC and CSE-on; the §9 "3.30×" is the gcc/size-1 starting
point. The cc-pVTZ §11 grid is a *different* problem (larger gaps, multi-rank). Any number measured with a
quarantined kernel on (`SPTC_SCALE_GEMM`/`SPTC_CE_E_GEMM`/aux-batch) or on the gcc/MKL toolchain is
non-canonical — see `docs/HARNESS_VS_EXPERIMENTS.md`. This section is the canonical single-node cold
cache-free comparison under the stock build.

## Level 1 (primary): per-operation isolated einsum suite — STATUS

The rigorous decomposition: run each individual contraction in isolation on identical COO operands, no reuse
(`mode=isolated_no_reuse`), normalized to **logical GFLOP/s**, checksum-gated — same op, same data, so it
isolates per-op execution across TA builds and separates the "keep-BLAS-fed" scheduling effect (which lives
only in the whole-residual, in-context measurement) from per-op cost.

**Harness (validated, this repo):** `src/ta_einsum_replay_main.cpp` + `src/einsum_bench_coo.h` +
CMake target `ta_einsum_replay` — builds against stock TA and computes correctly (verified: an L·R matmul
reproduces nnz/sum/sumsq/max_abs exactly), emitting the common `mpqc-einsum-results/v1` schema. It handles
all four families including the **ragged ToT** cases (`build_bench_nested` + `load_ragged_extents`) — the
CSV/PNO structure that flat NumPy einsum cannot express (`docs/NUMPY_BACKEND.md`). The MPQC side
(`mpqc-benchmark/einsum_bench/` + `bin/einsum-suite.py`) passes its 10/10 tests.

**Discovery is real:** an ethane Level-4 MPQC trace (this SIF) yields **305 products** across all four
families with MPQC's actual per-product timings, families, annotations, byte sizes, and nnz.

**Blocked on:** the panel `select`/`prepare-export` path requires ProductBench's serialized `ResultExpr`
fields, and ProductBench is **uncommitted work-in-progress** in `mpqc4` (`cck.ipp` modified; not in the
SIF's pinned commit `93593706ed`; the SIF binary has no "ProductBench" string). Completing Level 1 needs a
SIF rebuild from that WIP — see `docs/scaling-campaign-data/` notes / the campaign README. Once the rebuilt
SIF is available: `discover-log → select-panel → prepare-export → (MPQC export run) → build-suite →
run-einsum-suite.py (repo arm) + native-results (MPQC arm) → summarize`.
