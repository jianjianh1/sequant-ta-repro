# What MPQC does on a single thread — and the real single-thread lever

*Measured 2026-08-01 on node3 (Xeon D-1548, 8 core / 16 SMT), C2H6 cc-pVTZ CSV-CCSD cold T2,
1 thread, `SPTC_COARSE_OCC=9 OCC_TILE=2 COARSE_PAD=0 TILES_PER_DIM=8`, checksum-gated
(nnz=150301). Both sides link the identical TiledArray fork `cd53bd3e`. This document **corrects
`MPQC_RUNTIME.md` / `MPQC_ABLATION.md`**: the single-thread gap is neither thread-scheduling, nor
array compaction, nor the μ̃Κ contraction kernel — it is a class of half-transform terms that TA
executes as a per-cell scalar broadcast instead of a GEMM.*

## TL;DR

- **Baseline correction.** The historical "≈53 s" single-thread T2 was partly a gcc/MKL artifact.
  A clean **clang/OpenBLAS arena** build runs C2H6 cold T2 in **29.9 s** at 1 thread vs MPQC
  **8.26 s** → a **3.6×** gap (not 6.4×).
- **Compaction is NOT the lever (refuted by a clean same-binary A/B).** `compact_csv_coeffs`
  on/off = 29.9 vs 29.6 s, identical fallback counts. The repro's ToT leaves are *already*
  single-page (built up front via `arena_outer_init`), so compacting them is a no-op. Every prior
  round's "array-construction / compaction" hypothesis is wrong for single-thread.
- **The real root cause (new).** At 1 thread the #1 symbol is `fused_scale_t_x_tot_inplace` at
  **33.5%** (8-thread: repro 18% vs MPQC ~1%). It is **358.8 million per-cell scalar×vector AXPY
  calls** streaming 19.7 B elements. These come from ~69 flat×ToT terms — dominated by the μ̃Κ
  half-transform `I(i,i,μ̃,Κ;a) = Σ_μ̃ g(μ̃,μ̃,Κ)·C(i,i,μ̃;a)` (`generated_t2:487`) and its cousins
  — where the PNO index `a` rides as a **ToT inner spectator (broadcast)** and the contraction is
  over an **outer** index (μ̃/Κ). Because there is no inner contraction, TA's `BatchedContractReduce`
  cannot use the strided-DGEMM path and falls to a per-(output-cell × contracted-k) scalar AXPY.
- **What MPQC does instead.** The same math is a **per-pair GEMM**: for each occupied pair,
  `I[n,a] += g[n,k]·C[k,a]` (contract μ̃=k, with the PNO `a` as the GEMM free/N dimension). The GEMM
  reads `C[k,a]` once and reuses it across all `n=(μ̃',Κ)` via cache-blocking; the per-cell AXPY
  re-streams it for every `n`. That reuse is the whole difference — hence MPQC's ~1%.
- **Op-level ceiling (checksum-validated, scales).** `tools/gap_microbench` does the giant block
  both ways at 1 thread: C2H6 **TA 27.05 s vs hand per-pair GEMM 3.67 s = 7.4×**; C3H8 **115.9 vs
  15.9 s = 7.3×** — results matching to ~1e-13 (hand 14.4–15.5 vs TA 1.96–2.13 GFLOP/s). The lever
  is real, large, and size-stable.
- **LANDED (2026-08-01, `SPTC_SCALE_GEMM=1`, env-gated).** A batched scale-GEMM strided op in the TA
  arena einsum path takes C2H6 cold T2 from **30.60 → 20.01 s at 1 thread (1.53×)**, checksum exact
  (nnz=150301, sumsq/max_abs identical; sum differs at the ~13th digit = FP summation order). The
  358.8 M per-cell scalar AXPYs → **0** (all batched). Gap to MPQC 8.26 s: 3.7× → **2.4×**. Default
  (gate off) is byte-identical to baseline. Patch + reproduce: `singlethread/scale_gemm_patch/`.

## How the gap was localized (measurement trail)

| step | tool | result |
|---|---|---|
| baseline | residual, 1 thr, clang/OpenBLAS arena | cold T2 **29.9 s**, nnz=150301 ✓; MPQC 8.26 s |
| compaction A/B | `SPTC_COMPACT_COEFFS` 0 vs 1, same binary | 29.9 vs 29.6 s; ce+ce fallback 22736 both → **no effect** |
| GEMM accounting | `TA_GEMM_TIMING` | instrumented arena GEMM ≈ 3.5 s of 29.9 s; ce+ce "22736 fallback runs" |
| flat self-time | `perf -F 999` (1 thr) | **`fused_scale_t_x_tot_inplace` 33.5%**, dgemm 9.5%, `element_product_op` 1.9% |
| scale volume | custom `[scale-fused]` counters | **358.8 M** cell-calls, 19.7 B elems, max cell 63 (rank-1 inner) |
| caller | `perf --call-graph lbr` | `BatchedContractReduce::contract_pair` → per-cell scale op |
| site attribution | parse `generated_t2` | **69 `scale_right`** sites; dominated by μ̃/Κ half-transforms (L487 etc.) |
| ceiling | `gap_microbench`, 1 thr | TA 27.05 s vs hand-GEMM 3.67 s = **7.4×**, checksum match |

## Why TA can't GEMM these (and MPQC can)

The half-transform `I(i,i,μ̃',Κ;a) = Σ_μ̃ g(μ̃,μ̃',Κ)·C(i,i,μ̃;a)` has three index roles:
`a` (PNO) is the **inner** ToT mode and appears in the output (spectator/broadcast); μ̃ is an
**outer** mode and is **contracted**; (μ̃',Κ) are outer free. TA's ToT einsum sees "flat `g` ×
ToT `C`, no inner contraction" → `RegimeAInnerKind::scale_right` → `fused_scale_t_x_tot_inplace`
`result_cell(a) += g_scalar · C_cell(a)`, invoked once per (output cell × contracted-μ̃). The
strided-DGEMM fast path (`arena_strided_dgemm_ce_e`) requires an **inner** contraction to put in
BLAS's K dimension; here the contraction is outer and the inner is a spectator, so it never fires.

The same contraction is a GEMM if you treat the per-pair PNO `a` as the GEMM N dimension:
`I[(μ̃'Κ), a] = g[(μ̃'Κ), μ̃] · C[μ̃, a]` per occupied pair (`M=μ̃'·Κ, N=a, K=μ̃`). MPQC's evaluator/
array layout realizes exactly this per-pair GEMM (`gap_microbench` reproduces it and matches to
1e-13). The GEMMs are skinny (`a≈45–63`), so even the hand-GEMM is only ~5–10% of peak — but 7.4×
the per-cell AXPY, because BLAS reuses `C[μ̃,a]` from cache across the (μ̃',Κ) rows while the AXPY
re-streams it 19.7 B times from memory.

## The fix (landed) and its scope

Batch each `scale_right` SUMMA output cell's contracted-k run into one GEMM instead of k per-cell
AXPYs — a "flat-matrix × ToT-column-of-vectors → ToT-column" arena kernel (`arena_strided_scale`),
installed via the same `ContEngine` → `ContractReduce` strided-op hook the existing
`arena_strided_dgemm_ce_e` uses, env-gated (`SPTC_SCALE_GEMM`). Per occupied pair it gathers R's
μ̃-run of inner-a vectors into `Rmat[K×Q]` and issues one GEMM `tmp[M×Q]=L[M×K]·Rmat`, scattering
into result cells; non-uniform pairs fall back to the correct per-cell AXPY. This targets all 69
`scale_right` sites at once (`gap_microbench`'s per-pair GEMM was the validated reference/ceiling).

**Scaling / multi-thread (reframes the campaign).** The scale-GEMM helps at 1 thread (C2H6 1.53×,
C3H8 1.45×) and at 8 threads (C2H6 5.94 → 4.94 s, 1.20×), all checksum-exact. But the more important
finding is the thread-scaling contrast: **MPQC barely parallelizes (8.26 → 7.8 s, 1.06× over 8
threads) while the clang/OpenBLAS repro scales ~4×/8** (20.0 → 4.94 s with the fix). So MPQC wins at
1 thread (single-thread efficiency) but the **repro wins at 8 threads** — even the *baseline* arena
build (5.94 s) already beats MPQC's 7.8 s there. The historical "repro is slower" gap was (a) a
gcc/MKL vs clang/OpenBLAS toolchain artifact and (b) the single-thread fused_scale, now fixed; it was
never a whole-story deficit. (C4H10 segfaults on both baseline and treatment — a pre-existing arena
memory limit on its giant intermediate, not a kernel regression.)

**Measured (landed):** C2H6 cold T2 1-thread **30.60 → 20.01 s (1.53×)**, checksum exact. The
fused_scale share (~10 s) is removed. A perf profile of the fixed 1-thread run confirms fused_scale
is **gone** (was 33.5% → 0) and the new top is `dgemm_kernel` 18% + copies ~7% (the *actual* GEMM
work, comparable to MPQC's ~3 s of gemm), the `arena_strided_scale` kernel's own gather/scatter 6.7%,
COO load parsing ~15%, arena machinery ~8%, and the 26 `ce+ce` "non-canonical inner perm" reverts
only 2.5%. So **there is no single big remaining lever** — the μ̃Κ scale class was *the* lever and is
now closed; further single-thread gains are incremental (trim the kernel's scratch copies, reduce
arena machinery), not another clean 1.5×. (Single-node only — arena has a known np≥8 lazy-deletion
segfault; owning ToT ships for multi-rank. C4H10 segfaults on baseline *and* treatment: a pre-existing
arena memory limit, not a kernel regression.)

## Corrections to the prior record

- `MPQC_RUNTIME.md` "the single-node lever is MPQC's array-construction approach (compaction /
  build_tot_array rewrite)" — **refuted**: compaction is a no-op (leaves already single-page) and
  makes no single-thread difference; the lever is the scale-vs-GEMM dispatch of the broadcast-inner
  half-transforms.
- `MPQC_ABLATION.md` / `MPQC_RUNTIME.md` "same tiny ~60×60×18 per-pair GEMMs both sides; the edge is
  scheduling / per-op overhead" — the *contraction* GEMMs are indeed similar, but they are ~3.5 s of
  29.9 s; the dominant cost is the **non-GEMM** scale path that the GEMM census never counted.
- "thread-starvation" framing is an 8-thread artifact; at 1 thread there is no pool to starve and
  the cost is plainly the per-cell scale streaming.
