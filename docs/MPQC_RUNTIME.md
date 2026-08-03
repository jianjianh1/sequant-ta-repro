# How MPQC's runtime evaluates the residual — and why it beats the repro

*Measured on node3 (2026-07-31), C2H6, with TiledArray's own env-gated instrumentation
(`TA_STRIDED_DGEMM_VERBOSE`, `TA_GEMM_TIMING`) — the **same cd53bd3 TA fork** drives both MPQC
(the PaRSEC/Pthreads SIFs) and the repro. Companion to `MPQC_ABLATION.md`, which it **corrects**:
the single-node advantage is not "large batched GEMMs," it is scheduling of identical tiny GEMMs.*

> **UPDATE (2026-08-01) — a later diagnostic pass ("can you reproduce the runtime?") refines the
> body below and answers it. Thread sweeps show it is NOT concurrency/scheduling: MPQC's cold T2
> barely parallelizes (1→8 thr 8.26→7.80 s, 13%) while the repro parallelizes *better*; MPQC wins on
> **single-thread speed** (its entire T2 runs in 8.26 s on one thread vs the repro's *single giant op*
> at 26.5 s). Same tiny GEMMs — the repro just carries ~10× more per-GEMM execution overhead. So
> "scheduling to keep the threadpool fed" (below) is itself superseded by "far lower per-op execution
> overhead, single-thread"; and reproducing the EVALUATOR would not close the gap. See the final
> section "Can the repro reproduce the runtime? — measured verdict."**

> **CORRECTION (2026-08-03) — supersedes the body's single-node numbers, lever, and mechanism.** Three
> body claims are now refuted by later measurement; read them as historical:
> 1. **The "1-thread arena 54.2 s / ~6× gap" is a gcc/MKL toolchain artifact.** The honest
>    clang/OpenBLAS arena baseline is **29.9 s @1thr → 3.6×** (`MPQC_SINGLE_THREAD.md`), and at 8
>    threads the repro (5.94 s) **beats** MPQC (7.8 s) — there is no ~6× whole-story gap.
> 2. **The "array-construction / compaction / `build_tot_array` rewrite" lever is refuted.**
>    `SPTC_COMPACT_COEFFS` is a same-binary no-op (leaves already single-page); the real single-thread
>    lever is the scale-vs-GEMM dispatch of the broadcast-inner μ̃Κ half-transform, closed cheaply by the
>    landed `SPTC_SCALE_GEMM` (1.53×@1thr). See `MPQC_SINGLE_THREAD.md`.
> 3. **The mechanism is compute/dispatch-bound, not memory/construction.** `MPQC_PROFILE_DEEP.md` FC1
>    measured the kernel at IPC 2.55, DRAM 4-9 % of peak, L1-resident — the ~53 % "per-inner-cell
>    dispatch/construction" profile below reflects the *instruction count* of the per-cell path, which
>    the scale-GEMM cuts (274.9 B → 155.9 B), not an array-construction cost. See `MPQC_PROFILE_DEEP.md`
>    (unified verdict) and `MPQC_RUNTIME_EVAL.md` (the `sequant::evaluate` port was built and is 1.2×
>    slower — confirming reproducing the evaluator does not help).

## The question

`MPQC_ABLATION.md` measured that MPQC's cold residual is BLAS-bound (44–53% dgemm self-time) while
the repro's is thread-starved (6–8% dgemm, ~60% `ConditionVariable::wait`), and *inferred* the cause
was MPQC's "occ-as-batch layout coalescing the contraction into large batched GEMMs." This document
tests that inference directly by logging every GEMM both sides issue.

## Finding 1 — the GEMMs are identically tiny on both sides (the inference was wrong)

`TA_GEMM_TIMING` dumps the (M,N,K) of every strided-DGEMM the ToT contraction path issues. For the
μ̃Κ half-transform hotspot:

| side | dominant GEMM shapes (M×N×K) | ~total calls | GFLOP/s |
|---|---|---|---|
| **repro** (arena) | 63×63×18, 61×61×18, 50×50×18, 45×45×18 | ~170k | 6.6–10.5 (~3–4% peak) |
| **MPQC** | 61×61×14, 63×63×30, 50×50×14, 45×45×14 | ~170k | 2.9–3.6 (~1–1.5% peak) |

Both issue **~170,000 tiny per-pair GEMMs of ~60×60×~18** — M,N are the per-pair PNO count (45–63,
intrinsically small), K is a contracted-μ̃ tile. MPQC's are if anything *smaller and slower*. **GEMM
size is not the differentiator**, and MPQC does **not** batch occ into large GEMMs. Both also mostly
FIRE the same strided-DGEMM fast path (not a fast-path-vs-fallback difference). This **refutes the
`MPQC_ABLATION.md` "occ-as-batch → large batched GEMM" wording.**

## Finding 2 — the difference is scheduling overhead around identical GEMM work

The GEMM *kernel* time is essentially the same per residual pass (~3 s), and within the kernel both
are ~91–95% gemm. But that kernel is a very different fraction of the **wall**:

- MPQC: ~44–49% of wall in dgemm (perf self-time) — the pool stays in BLAS.
- repro: ~6% of wall in dgemm — ~94% of wall is *outside* the kernel, in MADNESS task-queue
  waiting (`ConditionVariable::wait`/`sched_yield`).

Same tiny GEMMs, same total GEMM work — MPQC keeps the 8 threads *busy* on them; the repro's pool
*starves* between them.

## Finding 3 — every "obvious" cause is ruled out empirically

| candidate | test | result |
|---|---|---|
| GEMM size / occ-batching | `TA_GEMM_TIMING` shapes (F1) | identical tiny GEMMs — **not it** |
| runtime cache/reorder (`cache_imeds`) | profile MPQC-nocache cold T2 | 18.7 s (2.3× slower) but **still 49% dgemm** — **not it** |
| task backend / threadpool | Tier 2: MPQC-**Pthreads** (native MADNESS pool) | also BLAS-bound (45% dgemm) — **not it** |
| MADNESS wait policy | repro busy-wait vs yield | 10.65 s vs 10.45 s (identical) — **not it** |

`cache_imeds` makes MPQC *faster* (less redundant work) but doesn't change that its pool stays fed;
the backend and wait policy are irrelevant on the same threadpool.

## Finding 4 — what's left: task-graph concurrency

The only remaining difference is the **task graph the evaluation driver builds over the identical
GEMM work**. MPQC's runtime evaluator, walking the SeQuant term forest and issuing one `TA::einsum`
per node over CSV-solver-tiled `ArenaTensor` operands (inner cells compacted to single-page
constant-stride, `mpqc4/.../mbpt/csv.ipp:44-64`; trange/pmap inherited from the CSV solver), exposes
enough ready parallel work to keep 8 threads in BLAS. The repro's **static generated einsum
sequence** (`src/generated_t2_residual.cpp:487-491`) over its own `build_tot_array` tiling
(`src/ta_builder.h`) exposes only ~42%-efficient parallelism (repro cold-T2 thread sweep
1/2/4/8 thr = 53.9/30.2/20.9/15.9 s, `gap_profile.txt`) — the pool idles waiting for the giant
μ̃Κ intermediate's few coarse tiles to reduce. Critically, the repro's tiling/pmap knobs
(`TILES_PER_DIM`, occ tiling, `SPTC_CYCLIC_PMAP`) **do not recover** this (ablation §11) — it is
structural to how MPQC's runtime tiles and drives the einsum, not a tunable.

## Source mechanism (three layers) — corrected interpretation

The source trace (over `mpqc4`, `sequant-fork`, `third_party/tiledarray-cd53bd3`) explains *how*
these task graphs arise — but note the corrected reading: these choices set **task granularity /
concurrency**, not GEMM size.

1. **Layout** (`mbpt/csv.h:102`, `pao_to_pno_mp2.ipp:510`, `eval_expr.cpp:117-130`): occ + expansion
   μ are outer/dense modes, the CSV virtual `a` the sole inner mode, inner cells compacted. This
   fixes how a contraction is split into outer tiles (→ reduce-tasks) vs within-tile batch loops
   (`dist_eval/contraction_eval.h:1211`, `tile_op/batched_contract_reduce.h:150`).
2. **TA dispatch** (`expressions/cont_engine.h`, `tensor/arena_einsum.h:1170`): task count =
   `my_slabs·local_size·k`; each task runs a batch loop of the tiny per-pair GEMMs. Same fast path on
   both sides — the difference is how many tasks and how much ready concurrency, set by the trange.
3. **Cache/reorder** (`sequant-fork/.../eval/cache_manager.hpp`, `optimize/sum.cpp`): CSE +
   t-independent persistence build each shared subnetwork once — a *work-reduction* (speed) lever,
   confirmed orthogonal to the BLAS-bound/starved split (Finding 3).

The eval-trace (`traces/checksum-run/ethane-checksum.log`) confirms MPQC forms the μ̃Κ intermediate
as one `Eval | Product` `g(i,i,Κ)*g(μ̃,μ̃,Κ)->I(i,i,μ̃,μ̃)` — the same single-array intermediate the
repro's `generated_t2:487` builds. Same contraction structure, same GEMMs; different scheduling.

## Conclusion

MPQC's runtime advantage on one node is **not** bigger/coalesced GEMMs — both sides issue the same
~170k tiny ~60×60×18 per-pair GEMMs (per-pair PNO is intrinsically small for everyone). MPQC's
runtime evaluator + CSV-solver array tiling **schedule** those identical GEMMs so the MADNESS
threadpool stays in BLAS; the repro's static generated sequence + self-built tiling expose only
~42%-efficient parallelism and the pool starves. This is not the GEMM size, the cache/reorder, the
backend, or the wait policy (all ruled out). Closing it needs the repro to reproduce MPQC's runtime
task-graph concurrency (its array tiling/pmap and evaluation dataflow), which its tuning knobs do not
reach — consistent with `MPQC_ABLATION.md`'s "reproduce the runtime evaluator" conclusion, now with
the precise reason: **it's the scheduling of identical tiny GEMMs, not their size.**

**Corrects `MPQC_ABLATION.md`:** replace "occ-as-batch coalescing into large batched GEMMs" with
"scheduling identical tiny GEMMs to keep the threadpool fed." (This "scheduling" framing is *itself*
refined below — see the measured verdict.) The gap's character and the top-line conclusion (it's not
GEMM size/backend/tile-type — it's how MPQC executes the residual) stand.

## Can the repro reproduce the runtime? — measured verdict (2026-08-01)

A cheap diagnostic pass (existing binaries + TA env instrumentation, no builds) answers this directly
and **refines the mechanism a final time**:

- **D2 — the giant op is slow/starved IN ISOLATION.** Profiling `tools/gap_microbench` (which isolates
  just the μ̃Κ op, generated_t2:487-488, then runs a serial hand-GEMM ceiling with *no* MADNESS tasks)
  shows `madness::ConditionVariable::wait` = 40.5% and ~few% dgemm for that one op alone. So the
  deficit lives in that single op's TA execution — not the evaluator (it calls the same einsum), not
  the per-einsum entry fence (`einsum/tiledarray.h:525`, within one op), not the ~250 small ops.
- **D3 + thread sweeps — it is NOT concurrency.** MPQC's cold T2 barely parallelizes (1→8 thr:
  8.26→7.80 s, **13% efficiency**); the repro parallelizes *better* (isolated op 5.0×/62%, whole T2
  3.4×/42%). The decisive same-node comparison at **1 thread**: MPQC's *entire* T2 residual runs in
  8.26 s while the repro's *single giant op alone* takes 26.5 s — **3.2× slower for a fraction of the
  work**, on the same tiny GEMMs (~3 s of actual gemm on both sides, E1/E2).

**Verdict: reproducing MPQC's runtime evaluator (`sequant::evaluate()`) would NOT close the gap.** The
evaluator issues the identical `TA::einsum` over the identical tiny GEMMs; the deficit is ~10× more
**per-GEMM TA/MADNESS execution overhead** in the repro's ToT einsum, seen cleanest single-thread. That
overhead is a property of the **array construction**, not the caller: MPQC's compacted single-page CSV
inner cells + CSV-solver `trange` (`mbpt/csv.ipp:44-64`, `pao_to_pno_mp2.ipp:510`) vs the repro's
`build_tot_array` ToT arrays (`src/ta_builder.h`, `src/ta_tensors.h`). It is not the evaluator, not the
entry fence, and not a JSON/tiling knob (the ablation showed knobs don't reach it).

**So the reproduction that would matter is of MPQC's array construction, not its evaluator** — build
the residual's ToT operands with compacted single-page inner cells and the CSV-solver's `trange` so
each strided-DGEMM carries less per-cell machinery. That is a structural change to `ta_builder.h` /
`ta_tensors.h`, feasible in-repo but non-trivial, and the definitive next experiment. (Open: whether
the dominant overhead is inner-cell compaction, the `trange` tile structure, or the arena-cell task
machinery is narrowed but not fully isolated — pinning it exactly would need MADNESS task-level
tracing via a compile-time `TA_TRACE_TASKS` rebuild.)

## Array-construction overhead — pinned (2026-08-01)

A cheap 1-thread pass (existing binaries, no builds) isolates *which* array-construction cost is the
~10× per-GEMM overhead. At 1 thread there is no idle-thread `ConditionVariable::wait` to mask it, so
the flat self-time symbols of the isolated giant op (`gap_microbench`, owning = the repro's shipped
ToT tile type) show the construction directly:

| category | ~% of 1-thread wall (load-excluded) | dominant symbols |
|---|---|---|
| per-inner-cell dispatch + construction | **~53%** | `std::_Function_handler<…Tensor<double>…>` 37% (one op per inner cell, ~170k cells), `arena_tot_grow_inplace`, `ContractionArenaPlan`, `transpose`, `arena_outer_init`, vector/emplace churn, malloc/free/memset ~6% |
| actual GEMM | **~19%** | `dgemm_kernel_HASWELL` + copies |

So **construction/dispatch is ~2.7× the actual GEMM.** And it is **per-cell, not per-tile**: sweeping
`TILES_PER_DIM` 4/8/16 barely moves the 1-thread time (54.6 / 53.7 / 60.8 s), consistent with the
ablation finding that tiling knobs don't close the gap.

**The mechanism:** the repro's owning ToT issues a `std::function`-wrapped operation **per tiny inner
cell**; MPQC's compacted-arena strided-DGEMM (`mbpt/csv.ipp:51-64` `compact_csv_coeffs` +
`arena_einsum.h` ce+e fast path) **batches** all cells of a contracted run into ONE BLAS call over a
single-page constant-stride arena — no per-cell dispatch, no per-tile arena rebuild. The repro's arena
build never compacts, and the repro switched to owning anyway (arena multi-rank segfault, §11).

**Verdict for reproducing MPQC's array construction:** the lever is **cell-batching / compaction**
(batch the per-pair PNO inner cells into a single strided BLAS call, as MPQC does), **not** fewer/bigger
outer tiles. Concretely that means porting `compact_csv_coeffs` and keeping the strided-DGEMM ce+e fast
path over an owning-stable arena (or teaching the owning ToT to batch its per-cell ops). Both are
TA-level changes entangled with the arena multi-rank segfault the repro abandoned arena for — feasible
but non-trivial, and the definitive next prototype if the gap is worth closing in-repo. Tuning knobs
(tiling, pmap), the evaluator, and the backend are all confirmed *not* the lever.

## Does the arena build already close it? — no (2026-08-01)

The pinned lever above suggested the repro's *arena* build (which batches cells into strided DGEMMs,
unlike the shipped owning build) might already close the single-node gap. A direct arena-vs-owning
comparison (existing binaries, checksum-matched, C2H6 cold T2) settles it. **All three sides were
re-measured with BLAS pinned to 1 thread and single-core execution CPU-verified** (see the
thread-fairness box below — the first-pass numbers had the repro's OpenBLAS unpinned while MPQC's was
pinned, an apples-to-oranges bias now removed):

| | arena (`build-cd53bd3`) | owning (`build-owning`, shipped) | MPQC |
|---|---|---|---|
| 8 threads (BLAS-pinned) | 9.42 s (1.21×) | 11.05 s (1.42×) | 7.8 s |
| 1 thread (BLAS-pinned)  | 54.2 s (6.2×)  | 47.9 s (5.5×)   | 8.75 s |

**At 1 thread, arena ≈ owning (~48–54 s; if anything owning is *faster*), both ~6× MPQC.** So the
batching does **not** help single-thread — the tile type is *not* the single-thread lever. The ~6×
single-thread gap is the **array construction** itself (`src/ta_builder.h build_tot_array`), common to
*both* tile types, vs MPQC's up-front-shaped + compacted, registry-tiled CSV-solver arrays. (This
refines the "per-cell dispatch" reading: owning pays a `std::function` per cell; arena avoids that
specific cost but carries equivalent `arena_outer_init` / multi-page construction overhead — net the
same 1-thread time.)

At 8 threads arena is ~15% faster than owning (9.42 vs 11.05 s; 1.21× vs 1.42× MPQC) because it
parallelizes slightly better — arena is usable single-node (an earlier 282 s reading was a transient
fluke). The repro ships owning only for **multi-rank** stability (the np≥8 arena lazy-deletion
segfault), not single-node speed.

### Thread-fairness verification (2026-08-01) — is MPQC's fast number *really* single-threaded?

MPQC links multithreaded OpenBLAS, so `MAD_NUM_THREADS=1` alone would not stop its DGEMMs from
grabbing cores. To rule out that MPQC's edge is hidden BLAS threading, each side was run at
`MAD_NUM_THREADS=1` under two BLAS configs — fully pinned (`OMP=OPENBLAS=MKL=1`) vs BLAS-unpinned —
while sampling the compute process's instantaneous CPU% (utime+stime over all threads) every 0.4 s:

| 1-thread run | cold T2 | residual CPU% (median / max) | peak threads |
|---|---|---|---|
| MPQC pinned    | 8.75 s | 108 / **112**  | 3  |
| MPQC blasfree  | 11.42 s| 108 / **222**  | 4  |
| arena pinned   | 54.2 s | 108 / **112**  | 3  |
| arena blasfree | 52.9 s | 110 / **1932** | 18 |
| owning pinned  | 47.9 s | 108 / **115**  | 3  |
| owning blasfree| 47.9 s | 110 / **2060** | 18 |

Two facts fall out. **(1) MPQC's fast single-thread number is genuinely single-core**: pinned it holds
at 108–112% CPU (one core plus a sliver of infra thread), and *unpinning* BLAS makes it **slower**
(11.4 s, CPU to 222%) — the per-pair GEMMs are too tiny to amortize thread spawn/sync, so multi-core
BLAS is counterproductive. The historical 8.26 s (run with `OMP_NUM_THREADS=1`) ≈ the pinned 8.75 s, so
it was already clean. **(2) The repro's ~50 s is likewise not a pinning artifact**: pinned ≈ blasfree
(the unpinned runs *do* fan BLAS out to ~19 cores, CPU spiking to ~2000%, yet wall time barely moves) —
direct proof the bottleneck is construction/dispatch, not GEMM compute. The ~6× single-thread gap
survives a strictly apples-to-apples, CPU-verified single-core comparison. At 8 threads, unpinned BLAS
*over*subscribes (25 threads on 16 cores) and hurts the repro, so the earlier blasfree 8-thread numbers
(arena 10.2 / owning 12.8) were pessimistic; BLAS-pinned they improve to 9.42 / 11.05.

**Final verdict:** switching to the arena build buys ~15% single-node but does **not** close the gap
(1.21–1.42× at 8 threads, ~6× at 1 thread — all BLAS-pinned and single-core-verified). Neither tile
type, nor compaction (which only rescues the arena path's minority multi-page runs and can't help the
tile-invariant 1-thread gap), nor any knob, nor hidden thread count is the lever. The single-node gap
is MPQC's **array-construction approach** — a substantial rewrite of `build_tot_array` to shape ToT
arrays up front and compact them the way the CSV solver does — not a cheap in-repo change. This closes
the "reproduce the runtime" line of investigation: the gap is real, measured fairly, its lever is
precisely located (array construction, single-thread), and closing it is a construction-rewrite
project, consistent with every prior round's "not cheaply closable in-repo."
