# Turning off MPQC's features: what actually makes it faster than the repro

*Measured, not inferred (2026-07-31, node3 = Xeon D-1548, 8c/16t). Companion to
`GAP_RESEARCH.md`, which decomposed the gap by **source-reading**; this document
**turns MPQC's optimizations off one at a time and profiles** the residual, so the
attribution is empirical. It corrects one load-bearing claim in `GAP_RESEARCH.md`
(that MPQC shares the repro's thread-starved kernel — it does not).*

## Why

`GAP_RESEARCH.md` concluded, from reading the source, that MPQC and the repro run the
*same* `TA::einsum` on the *same* TiledArray and therefore hit the *same* thread-starved
skinny-GEMM ceiling single-node, so MPQC's only real lever was the PaRSEC backend
multi-node. That reasoning was never checked against a profile of MPQC itself. This study
does three things the prior work did not: (1) profiles **MPQC's** cold residual directly,
(2) **ablates** MPQC's runtime features (via its input JSON) to see which ones matter, and
(3) rebuilds MPQC with the **PaRSEC backend removed** to test the backend claim head-on.

## Method

- MPQC = the instrumented fork SIF (`mpqc-latest.sif`, PaRSEC) and a purpose-built
  `mpqc-pthreads.sif` (`MADNESS_TASK_BACKEND=Pthreads`, identical fork/ref). Repro = the
  generated `TA::einsum` sequence (`ta_sequant_native_residual_main`, owning-ToT; and the
  default arena build `build-cd53bd3`).
- Timing = `Eval | WholeResidualWallTime | R=2` (occ 1 = cold, occ 2 = warm). Correctness
  gate = `WholeResidualChecksum` occ 2 (nnz + sumsq + max_abs robust to ~10 digits; the
  signed `sum` is gauge/reassociation-sensitive and is not the gate).
- Profiles mirror `gap_profile.txt` exactly: `perf record -F 999 -g`, **flat self-time**
  (`--sort=dso` / `--sort=symbol`, no call graphs — Release binaries, no frame pointers).
  The MPQC cold-T2 residual is isolated by running `max_iter=2` and attaching `perf` to the
  `mpqc` PID immediately after the cold T1 residual prints.
- Ablations flip **one** CCk JSON key each (`ablation/make_ablations.py`); the approximation
  knobs `csv:tIJ`/`csv:tTA` are held fixed so the comparison is fair.

## Finding 1 — MPQC's cold residual is BLAS-bound; the repro's is thread-starved

Same node, same session, same perf recipe, cold T2 self-time:

| cold T2 (C2H6) | dgemm (`libopenblas`) | thread-sync | verdict |
|---|---|---|---|
| **repro** (owning-ToT) | 7.96% | `ConditionVariable::wait` 28% + kernel 31% ≈ **60%** | thread-starved |
| **repro** (arena) | 6.24% | ~28% kernel + binary sync | thread-starved |
| **MPQC** (PaRSEC) | **43.68%** | `MutexWaiter::wait` 2.8% + kernel 11% ≈ **14%** | BLAS-bound |

The character **scales**: MPQC C3H8 cold is 52.85% dgemm. This **refutes** the inferred
"MPQC hits the same thread-starved kernel." MPQC drives the identical einsum but keeps the
GEMMs coalesced and the threadpool fed; the repro fragments the μ̃-contraction into a huge
number of tiny per-pair tasks and spends ~60% of its wall synchronizing.

## Finding 2 — the only two features that matter, the repro already has

C2H6 ablation (turn OFF one MPQC feature; WallTime vs base 7.88 s cold / 3.71 s warm; all
safe variants pass the checksum gate):

| feature off | cold T2 | warm T2 | reading |
|---|---|---|---|
| `cache_imeds` | 17.2 s (2.2×) | **23.6 s (6.4×)** | dominant lever: ReorderSum + CacheManager. Off ⇒ warm collapses below cold. Lever grows with size (C4H10 warm **12×**). |
| `seq_opt` | **OOM/killed >1200 s** | — | factorization non-negotiable (>150×). |
| `opt_for=memsize` | 4.1 s (0.5×) | 33.6 s (9×) | cold-optimal, warm-catastrophic. |
| `n_replay=1` | 4.2 s (0.5×) | 15.2 s (4×) | same trade-off, milder. |
| occ/uocc tiling | 7.5–10.6 s | 3.5–15.7 s | minor (fine occ hurts, coarse neutral). |
| `batch:aux=96` | 8.3 s | — | neutral ⇒ aux-batch is memory-only, not speed. |

Two conclusions. (a) The two features that move MPQC — `cache_imeds` and `seq_opt` — are
exactly the ones the **repro already replicates** (its t-independent precompute and the same
SeQuant `optimize()`). (b) MPQC's default cost model is deliberately **warm-tuned**:
`memsize`/`n_replay=1` win cold but wreck warm, so the shipped default optimizes the steady
state that dominates a real multi-iteration CCSD run.

## Finding 3 — the inner-tile layout is a red herring

Hypothesis: MPQC keeps BLAS fed because it uses `ArenaTensor` inner tiles (coalesced) while
the repro adopted owning `Tensor` (fragmented). **Refuted by measurement:** the repro built
with the *same* `ArenaTensor` (`build-cd53bd3`) is *still* thread-starved (6.24% dgemm,
`__sched_yield` in the hot path) — it does not behave like MPQC. Arena vs owning is not the
separator.

## Finding 4 — single-node, the backend is irrelevant; it is the runtime evaluator

Rebuilt MPQC with PaRSEC removed (`mpqc-pthreads.sif`, banner `type = Pthreads`, checksums
match to 11 digits). Single-node:

| cold T2 | PaRSEC | Pthreads | dgemm% (PaRSEC / Pthreads) | repro |
|---|---|---|---|---|
| C2H6 | 7.88 s | 7.52 s | 43.7% / 45.0% | 10–12 s |
| C3H8 | 31.6 s | 31.0 s | 52.9% / 51.7% | 66 s |
| C4H10 | 77.0 s | 76.7 s | — | 155 s |

MPQC is BLAS-bound and equally fast with **either backend** (Pthreads marginally faster — no
PaRSEC overhead single-node). Crucially, the repro uses the **same native MADNESS
threadpool** as MPQC-Pthreads, yet is thread-starved and ~2× slower. That removes the last
confound: with backend, TiledArray, threadpool, and molecule all held equal, the *only*
difference is **static generated einsum (repro) vs runtime ReorderSum + CacheManager
coalescing (MPQC)**. The single-node gap is the evaluator's work-coalescing (keeping BLAS fed).
*(Correction 2026-08-03: the "~2× single-node" figure here used the gcc/MKL-era repro (12.4/66/155 s
np1); with clang/OpenBLAS the 1-thread gap is 3.6× and the repro **beats** MPQC at 8 threads (5.94 vs
7.8 s) — the residual single-node gap was toolchain + the now-fixed `fused_scale` scale-vs-GEMM
dispatch. See `MPQC_SINGLE_THREAD.md` (scale-GEMM landed) and `MPQC_PROFILE_DEEP.md` (compute-bound,
not memory). A naive `sequant::evaluate` port is 1.2× slower — `MPQC_RUNTIME_EVAL.md`.)*

*(Correction — see `MPQC_RUNTIME.md`.* Direct GEMM-level instrumentation later refuted the "large
batched GEMMs" reading: both MPQC and the repro issue the **same ~170k tiny ~60×60×18 per-pair
GEMMs**. MPQC's edge is that its runtime **schedules** those identical tiny GEMMs so the threadpool
stays in BLAS, while the repro's static sequence starves it — same GEMM work, different task-graph
concurrency. A runtime-evaluator/array-tiling property, not GEMM size and not a JSON knob.)*

## Finding 4b — the warm residual is a different, less-BLAS regime

Profiling the *warm* T2 (R=2 occ2, cache warm) shows dgemm drops to **24%** (from cold's 44%),
with more TA machinery (`ContEngine` 15%) and sync (`MutexWaiter::wait` 8.7% vs cold 2.8%). Warm
reuses the cached giant μ̃Κ intermediate — the big coalescible GEMM — leaving only the ~250 small
t-dependent ragged-ToT ops (lower arithmetic intensity). That is why the **warm gap (~2×) is
smaller than the cold gap**: less coalescible BLAS work for MPQC's evaluator to exploit.

## Finding 5 — multi-node: the backend is a MINOR lever (corrects prior claim)

Ran C4H10 on both SIFs across the fleet (np{1,4,8}; the testbed allowed 9 nodes after a manual
apptainer deploy — the fleet was apt-locked):

| np | PaRSEC cold | Pthreads cold | PaRSEC scaling | Pthreads scaling | PaRSEC/Pthreads |
|---|---|---|---|---|---|
| 1 | 77.0 s | 76.7 s | — | — | 1.00× |
| 4 | 13.6 s | 15.8 s | 5.7× | 4.9× | 1.16× |
| 8 | 11.1 s | 12.5 s | 6.9× | 6.1× | 1.12× |

**MPQC-Pthreads scales nearly as well as MPQC-PaRSEC** (6.1× vs 6.9× at np=8); PaRSEC's edge is a
steady ~1.1× that does **not grow** with np. This **corrects `GAP_RESEARCH.md`'s "PaRSEC is the one
lever that matters."** The repro's poor multi-node scaling (1.9–3.5×/16 in §11) is **not** the
Pthreads backend — MPQC on the *same* native threadpool scales ~6×/8. The repro scales poorly for
the same reason it is slow single-node: its **static-einsum evaluator**. (Caveat: reached np=8, not
np=16 — the decisive high-rank point wasn't provisionable here; but the np4→np8 trend shrinks, not
grows, arguing against a large divergence at np=16.)

## Conclusion

- The repro-vs-MPQC **single-node** gap (~2×) is **entirely MPQC's runtime evaluator**
  coalescing the residual into BLAS-bound work — not the backend (Finding 4), not the
  inner-tile layout (Finding 3), and not a feature the repro lacks (Finding 2, it has the
  two that matter). Matching it needs the repro to reproduce MPQC's runtime
  memoization/occ-batch coalescing, which a static generated einsum cannot.
- The **multi-node** gap is *also* the evaluator, not the backend (Finding 5): MPQC-Pthreads
  scales ~6×/8 — nearly as well as MPQC-PaRSEC — so the repro's poor scaling (1.9–3.5×/16) is its
  static einsum, not its Pthreads backend. PaRSEC adds only ~1.1×.
- Two prior conclusions are corrected: (1) `GAP_RESEARCH.md`'s "MPQC hits the same thread-starved
  skinny-GEMM kernel" is **wrong** — MPQC is BLAS-bound (44–53% dgemm) vs the repro's 6–8%; and
  (2) "PaRSEC is the one lever that matters" **overstates it** — the backend is a ~1.1× factor;
  the evaluator dominates both single- and multi-node.
- **Net:** to close the gap, the repro must reproduce MPQC's *runtime* evaluator (ReorderSum +
  CacheManager memoization + occ-batch coalescing) — a generator/backend project. Tuning knobs
  (tiling, pmap, aux-batch, proto extent), the inner-tile layout, and the task backend are not the
  lever. This is consistent across every experiment here.
