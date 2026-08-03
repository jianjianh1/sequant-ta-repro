# Fact-check of the doc set — 2026-08-03 full pass

*Supersedes the 2026-07-30 pass (5 docs / 171 claims; retained in git history). Scope: **all 16
documents** — `README.md`, `tools/README.md`, `docs/{MPQC_COMPARISON, MPQC_EVALUATION, MPQC_SINGLE_THREAD,
MPQC_MULTIRANK, MPQC_ABLATION, MPQC_RUNTIME, MPQC_RUNTIME_EVAL, MPQC_PROFILE_DEEP, GAP_RESEARCH,
CONTRACTION_IR, LAYERED_IR}.md`, `docs/scaling-campaign-data/{README,SUMMARY}.md`. Method: three parallel
audit agents verified every number / `file:line` citation / causal claim / cross-doc reference against (1)
the code (`src/`, `CMakeLists.txt`, generated files, the TA fork + `mpqc4`/`sequant-fork` trees), (2) the
committed data (`scaling-campaign-data/*.csv`, `profiles/`, `*.ctir`), and (3) the latest measured findings
(the 08-01/02/03 profiling work). Precedence: code/data is ground truth; a doc claim contradicting it is a
finding and the **doc** was corrected (no data was changed). Every finding cites concrete evidence.*

## Summary

- **~250 claims checked across 16 docs; the large majority clean** (all §11 scaling tables reproduce from
  `results.csv`/`mpqc_mr.csv`/`gap_decomposition.csv`; all CTIR figures match the `.ctir`; the load-bearing
  `mpqc4`/`sequant-fork`/TA citations resolve). **`MPQC_RUNTIME_EVAL.md` and `MPQC_MULTIRANK.md` were
  already accurate** (RUNTIME_EVAL clean; MULTIRANK needed only 2 annotations).
- **Four cross-doc corrections applied** (below). None changes the campaign's substance — repro and MPQC run
  the same algebra on the same TiledArray, the correctness anchors hold, the headline gap direction (MPQC
  faster at equal ranks) stands. The corrections make the *mechanism* prose match what was later measured.
- The two 2026-07-30 findings that had never been applied (**B1** ~87 %/~100×; **B2** proto=45 11.6 s) are
  now resolved (B1 corrected everywhere; B2 left as a labelled arena-era figure).

## The four cross-doc corrections (theme → docs → fix)

1. **"memory-bound / re-streams C ~19.7 B times" → compute/dispatch-bound.** `MPQC_PROFILE_DEEP.md` FC1
   *measured* the μ̃Κ kernel at IPC 2.55, DRAM 4-9 % of peak, L1-resident; the `SPTC_SCALE_GEMM` win is
   cutting instructions 274.9 B→155.9 B, not locality. Fixed in **MPQC_SINGLE_THREAD.md** (the source
   mechanism, ~L20/29/68) and **GAP_RESEARCH.md** ("memory/latency-bound"→"latency/overhead-bound"). No
   other doc asserted memory-bound *speed* (the "memory wall" elsewhere is legitimate RSS/OOM capacity).

2. **"one contraction ~87 % of cold T2 / ~100× off peak / per-outer-cell overhead of the flat×ToT op :487"
   → op :488 (ToT×ToT), ≈40 %, ≈2.5× ceiling, thread-starvation.** Refuted by committed `gap_profile.txt`
   ("top op = `:488`", "line 487 (flat×ToT) NOT in top ops", 30.5/77.75 s ≈39 %), `gap_ceiling.csv`
   (2.18-2.59× achievable, hand-GEMM ~9-10 % of peak), and `MPQC_PROFILE_DEEP.md` (compute-dispatch-bound,
   dgemm ~6 % self-time, ~45 % condvar-wait @8thr). This is old finding **B1**, never applied. Fixed in
   **MPQC_COMPARISON.md §11 (lever 1)**, **MPQC_EVALUATION.md** (correction box + §8 retitle),
   **CONTRACTION_IR.md**, **LAYERED_IR.md**. (:487 remains correctly described as the highest-cell
   CELL-BOUND node and the *1-thread* scale-GEMM target.)

3. **toolchain-inflated single-node numbers ("54 s / ~6×", "array-construction/compaction is the lever").**
   The honest clang/OpenBLAS baseline is 29.9 s @1thr (3.6×) and the repro **beats** MPQC at 8 threads
   (5.94 vs 7.8 s); compaction is a no-op and `SPTC_SCALE_GEMM` is the real 1-thread lever
   (`MPQC_SINGLE_THREAD.md`). Correction boxes added to **MPQC_RUNTIME.md** (the most-superseded doc),
   **GAP_RESEARCH.md** (Finding 4), **MPQC_ABLATION.md** (Finding 4 "full stop").

4. **runtime evaluator framed as a purely-open "generator/backend project" → cross-link the built port.**
   `MPQC_RUNTIME_EVAL.md` built the `sequant::evaluate` port and measured it **1.2× slower** (crash during
   bring-up was a leaf c1/c2 mis-map, not an evaluator bug). Cross-links added in **MPQC_MULTIRANK.md** (§6),
   **MPQC_COMPARISON.md** (§11 see-also), **MPQC_EVALUATION.md**, plus the correction boxes above.

## Per-doc verdict

| Doc | ~claims | findings | resolution |
|---|---|---|---|
| MPQC_RUNTIME_EVAL.md | 20 | 0 | clean (numbers self-consistent + match CSVs) |
| MPQC_MULTIRANK.md | 30 | 2 | arena-np2 diagnostic labelled; runtime-eval cross-link added |
| MPQC_SINGLE_THREAD.md | 15 | 3 | memory-bound mechanism corrected; 8thr fused_scale 18→11 %; 1-thr ceiling caveated |
| MPQC_PROFILE_DEEP.md | 30 | 3 | CSV-rounding: 5-9→4-9 %, FC2 syscall 11→9.2 %, FC3 dgemm ~10→6.9 % |
| GAP_RESEARCH.md | 25 | 4 | 08-03 box: memory→overhead-bound, toolchain single-rank term, scale-GEMM landed |
| MPQC_ABLATION.md | 15 | 1 | Finding 4 "~2×/full stop" softened + toolchain/scale-GEMM cross-link |
| MPQC_RUNTIME.md | 20 | 4 | 08-03 correction box (54 s/6× toolchain, compaction refuted, compute-bound) |
| MPQC_COMPARISON.md | 60 | 4 | headline scope caveat; §11 lever 1 (87 %/100×) corrected; see-also + materialise reconciled |
| MPQC_EVALUATION.md | 50 | 4 | 08-03 correction box; Stage-8 retitled "empirically refuted" |
| CONTRACTION_IR.md | 15 | 2 | 87 %/100× → op :488 ≈40 %/2.5× ceiling; §8 attribution superseded |
| LAYERED_IR.md | 14 | 2 | 87 % superseded note; `generator.hpp:39`→`:40` |
| README.md | 20 | 1 | arena-build vs `MAD_NUM_THREADS>1` note (use owning for 8-thread) |
| tools/README.md | 8 | 1 | added `gap_microbench`; "the C++ one"→"ones" |
| scaling-campaign-data/README.md | 12 | 2 | data list + ~13 missing CSVs; "4.5–7× at np16"→"4.5–5.5× (7× at np4)" |
| scaling-campaign-data/SUMMARY.md | 20 | 1 | "~5.4× over 16 ranks" disambiguated (repro self-speedup, not the gap) |
| docs/FACT_CHECK.md | — | — | this rewrite (was 5-doc/2026-07-30 scope) |

## Old-ledger reconciliation (2026-07-30 findings)

- The 27 upheld 07-30 findings were spot-checked as **applied** (A1 `339→15`→391.5→112 s; A3 traced-einsum;
  C1/C2/C3 EVAL rewordings; C5/C6/C8 SUMMARY; D8/D9/D10/D12 citations/CSE37) — all present in the current docs.
- **B1** (~87 %/~100×, never applied) — now **resolved** (correction #2 above), and additionally *contradicted*
  by the newer committed `gap_profile.txt`/`gap_ceiling.csv`, so it is corrected rather than merely caveated.
- **B2** (`proto=45 11.6 s` / `18.9 s` np1 not in a CSV, `MPQC_COMPARISON.md:376`) — left in place as an
  explicitly arena-era generator-extent figure; not reconciled to a committed cell (a superseded lever).

## Verification of this pass

`grep -riE "memory[- ]bound|re-stream" docs` now returns only refutation/correction discussion, never a live
assertion. The corrected numbers (op :488 ≈40 %, ≈2.5× ceiling, 29.9 s/3.6×, 1.2× slower, DRAM ≤23 % peak)
trace to committed CSVs (`gap_profile.txt`, `gap_ceiling.csv`, `hwcounters.csv`) or blessed measured findings.
No `src/` or committed-data file was modified — documentation only.
