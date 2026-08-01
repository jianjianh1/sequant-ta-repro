# The multi-rank gap: what it is, why every lever is refuted, and the honest verdict

*2026-08-01, node16-31 (Xeon D-1548, 1 rank/node, 8 threads/rank), clang/OpenBLAS owning-ToT vs the
instrumented MPQC SIF. All numbers cited from committed data (`scaling-campaign-data/{mpqc_mr.csv,
results.csv,gap_decomposition.csv,parsec_experiment.csv,pmap_distribution_C4H10_np16.txt}`). This
document characterizes the DISTRIBUTED gap and records why it is not overnight-closable — so the next
effort starts from an accurate map instead of re-testing refuted levers. Companion to
`MPQC_COMPARISON.md §11` (whose tables remain correct) and `MPQC_SINGLE_THREAD.md` (single-node, solved
this session).*

## TL;DR / verdict

Single-node is solved (the `SPTC_SCALE_GEMM` kernel + clang/OpenBLAS put the repro at parity/ahead of
MPQC at 8 threads). **Multi-rank is not.** The fair same-np cold gap is 4.3–7.0× for real molecules and
**the dominant term grows with molecule size** (the scaling term, not the per-op term). Every backend
and knob lever is refuted, and this session's single-node win (the scale-GEMM) **does not transfer to
multi-rank**. The residual gap is MPQC's **runtime evaluator** (ReorderSum + CacheManager coalescing)
plus its solver-inherited balanced layout — a generator/evaluator project, out of overnight scope.

## 1. The current gap (clang/OpenBLAS owning — not toolchain-stale)

Cold T2, fair **same-np**, ratio = repro / MPQC:

| mol | np1 | np4 | np8 | np16 |
|---|---|---|---|---|
| C2H6 | 1.6× | 2.7× | 2.0× | 1.6× |
| C3H8 | 2.1× | 4.3× | 4.8× | 4.5× |
| C4H10 | 2.0× | **7.0×** | 5.5× | 4.7× |
| C5H12 | 1.7× | 5.8× | 5.3× | **5.5×** |

Raw anchors (T2 s): repro C4H10 np1→16 = 155.5→48.9, MPQC 76.9→10.31. C5H12 repro 391.5→111.7, MPQC
233.1→20.47. (The multi-rank binaries are clang-21/OpenBLAS owning-ToT — verified in the campaign ELFs —
so this is **not** the ~1.8× gcc/MKL artifact that inflated the old single-node baseline; re-baselining
does not help.)

## 2. Decomposition — the scaling term dominates and grows (`gap_decomposition.csv`)

np=16 gap = (single-rank per-op gap) × (multi-node scaling gap):

| mol | single-rank gap | scaling gap | repro scaling | MPQC scaling |
|---|---|---|---|---|
| C2H6 | 1.60 | 0.98 | 1.87× | 1.84× |
| C3H8 | 2.12 | 2.11 | 2.18× | 4.59× |
| C4H10 | 2.02 | 2.35 | 3.18× | 7.45× |
| C5H12 | 1.68 | **3.25** | 3.50× | **11.38×** |

The per-op term is ~1.6–2.1× and flat; the **scaling term grows to 3.25× at pentane**, where MPQC
scales 11.4×/16 and the repro only 3.5×/16. Closing the multi-rank gap = closing the *scaling* term.

## 3. NEW (this session): the scale-GEMM does not transfer to multi-rank

The single-node lever (`SPTC_SCALE_GEMM`, `arena_strided_scale`) batches op-487's per-cell AXPYs into
one GEMM per occupied pair — 1.53× @1thr, 1.20× @8thr single-node. **At multi-rank it is slower:**

| C3H8 cold, np=2 | baseline | SPTC_SCALE_GEMM=1 |
|---|---|---|
| whole-T2 wall | 50.8 s | **70.8 s** (1.4× slower) |

Checksum-exact (nnz=261914). The optimized kernel (thread-local scratch, no per-call copy) only trimmed
83.7→70.8 s — it is not a per-call-overhead artifact. **Mechanism:** the distributed SUMMA splits the
contracted μ̃ into small K-panels, one per `strided_oprod_op` call, so each batched GEMM is tiny and
loses to the per-cell AXPY. The scale-GEMM's benefit is inversely related to distribution — a
single-node lever. (It is also arena-only, and arena segfaults at C4H10 np≥8.) So the per-op term of
the decomposition is **not** recoverable at multi-rank with this kernel.

## 4. Every distribution lever is refuted — and precisely why

| lever | result | why it fails |
|---|---|---|
| **PaRSEC backend for the repro** | 4.6–64× *slower* (`parsec_experiment.csv`; C4H10 np8 743 s vs Pthreads 64 s) | PaRSEC's per-task scheduling overhead is pathological on the repro's millions of tiny ToT tasks; appears even single-node (np1). MADNESS-Pthreads handles fine-grained shared-memory tasks far better here. |
| **`SPTC_CYCLIC_PMAP` / operand pmap** | no effect (`cyclic_pmap_timing.csv`) | `TA::einsum` re-derives the contraction distribution from the 2-D ProcGrid and **discards operand pmaps** (`cont_engine.h:830-952`; stated in the fork CLAUDE.md). Setting an input pmap cannot change how the μ̃Κ SUMMA distributes. |
| **rank co-location (the starvation)** | real but not a lever | `pmap_distribution_C4H10_np16.txt`: TA's default blocked pmap starves ranks 0,1,2 (0 tiles) on every *sparse* array, because frozen-core-zero leading tiles map to low ordinals. But einsum ignores operand pmap (row above), so fixing it does not rebalance the SUMMA. |
| **comm/compute overlap (backend)** | not it | MPQC rebuilt with PaRSEC removed (`mpqc-pthreads.sif`, same MADNESS-Pthreads single-progress-thread runtime) still scales 6.1×/8 vs PaRSEC's 6.9×/8 (`MPQC_ABLATION.md` Finding 5). The funnel exists for both sides; PaRSEC buys a steady ~1.1×, not the growing gap. |
| **proto=100 generator extent** | superseded | owning-ToT makes the giant intermediate cheap; proto=45's intermediate distributes across ranks *better* (C2H6 cold np16 6.6 vs 13.8 s). Also fails hexane (heap corruption). |
| **cross-term CSE / cache_imeds / seq_opt / sparse-threshold / R2-sym** | wall-negligible or already-present | the repro already has the warm/cold split (= cache_imeds) and the same SeQuant optimize(); CSE dedup finds only 7 redundant einsums, none touching the hotspot. |

## 5. What MPQC does that the repro does not

Same TA, same `TA::einsum`, byte-identical distribution algorithm. MPQC's advantage is its **runtime
evaluator** — ReorderSum + CacheManager coalescing that keeps the threadpool BLAS-bound (MPQC 44–53%
dgemm self-time vs the repro's 6–8%) — and its **solver-inherited layout**: the residual's DF/CSV
arrays inherit the CSV solver's `TiledRange`/`SparseShape`/pmap (`cck.ipp:1563-1565`), so the SUMMA of
the giant μ̃Κ half-transform is load-balanced across ranks *for free*. The repro emits a flat static
generated einsum over self-built tilings with TA's default pmap. Matching this is a generator/evaluator
change, not a knob.

## 6. Verdict

The multi-rank gap is dominated by the scaling term, which grows with molecule size. It is **not**
the backend (PaRSEC refuted, backend-neutral), **not** the operand pmap (einsum discards it), **not**
per-op compute (the scale-GEMM, the one new per-op lever, actively hurts at multi-rank). It is MPQC's
runtime evaluation + solver-inherited balanced layout. **Overnight-closable: no.** The path forward is
to reproduce that evaluation (coalescing + up-front balanced layout upstream of einsum) — a substantial
generator/backend project, explicitly out of this scope.
