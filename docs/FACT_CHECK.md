# Fact-check of the MPQC comparison doc set

*Generated 2026-07-30. Scope: `README.md`, `docs/MPQC_COMPARISON.md`, `docs/MPQC_EVALUATION.md`,
`docs/CONTRACTION_IR.md`, `docs/scaling-campaign-data/SUMMARY.md`. Method: 171 claims
(citations, numbers, cross-doc, causal) verified against the on-node MPQC/SeQuant source
(`/users/jianjian/mpqc4`, `/users/jianjian/sequant-fork`), the committed CSVs / `.ctir` files,
and fresh re-measurement on this node + a node16–31 re-sweep. Every finding below was
independently re-checked (adversarial pass) before being listed.*

## Summary

- **171 claims checked; 141 clean; 27 findings upheld; 3 candidate findings refuted.**
- No claim invalidates the comparison's substance: the repro and MPQC do run the same algebra on
  the same TiledArray (`cd53bd3`), the correctness anchors hold, and the headline gap direction
  (MPQC faster at equal ranks) is supported by the data.
- The defects are: a partial-refresh straggler (aux-batching marked done in one place, "to-do" in
  another), a number typo (`339→15 s`), several line-drifted citations, two figures with no
  committed backing (`~87%/~100×`, `proto=45 11.6 s`), ratio-direction/scope ambiguities, and — the
  biggest — a **now-stale hexane conclusion**: repro hexane is completing on regenerated leaves as
  this is written (see Re-measurement).

Severity legend: **WRONG-FACT** (states something false) > **UNSUPPORTED** (no backing evidence) >
**DRIFT** (self-inconsistent / stale after refresh) > **STALE-CITATION** (line-drifted or wrong
file) > **NIT**.

### Status of fixes (2026-07-30, applied during an active concurrent edit by the author)
- **Applied by this pass:** all EVAL citation line-fixes (D1–D6), the EVAL `ta_builder.h:363`
  wrong-fact + intro pmap framing (A2, C2), byte-identical wording (C3), warm-1.3× scoping (C7),
  lever (b) → DONE (C1), ~87%/§3 cross-ref (B1); the §11 CSV sync (E2); SUMMARY.md C5/C6/C8 + hexane.
- **Left to the author (was live-editing MPQC_COMPARISON.md — not touched to avoid clobbering):**
  A1 (`339→15 s`, partially done→`391.5→112 s`), A3 (`sptc_traced_einsum` path/env), D7–D9
  (result.hpp / generated_t2 citations), C4 (headline np16 range), C9 (t-indep count reconciliation),
  D10 (CSE37 uses=2), D11 (intro mpqc np16). All specified below — apply at will.
- **Needs an author decision (not auto-applied):** B1 (commit the per-op trace, or keep the
  "not committed" caveat), B2 (which run produced proto=45 11.6 s), C8/proto-divergence (~0.2% vs 7%).

---

## A. WRONG-FACT (act first)

### A1. `339→15 s` — dropped digit (MPQC_COMPARISON.md:343)
Claim: "the repro *does* scale (C₅H₁₂ cold 339→15 s over the sweep)". `results.csv` C₅H₁₂ cold T2:
np2=339.0, np4=215.1, np8=149.3, np16=114.8 — the minimum is **~115 s**, not 15 s. "15" is a
dropped leading "1".
**Fix:** `339→115 s` (committed np2→np16) or `391.5→112 s` (fresh, matching the np1=391.5 s cited
at line 335).

### A2. `ta_builder.h:363` "no explicit pmap" — wrong line and wrong fact (MPQC_EVALUATION.md:28, 197, 320, 418)
Claim: `ArrayToT array(world, outer_trange, sp_shape)` at `ta_builder.h:363`, "no explicit pmap
argument, so TA assigns its default." Reality: line 363 is comment prose; the construction is at
**ta_builder.h:430-431** and **does** pass a 4th pmap argument:
`ArrayToT array(world, outer_trange, sp_shape, maybe_cyclic_pmap(world, tiles_range.volume()))`.
`maybe_cyclic_pmap` (:146-151) returns empty (TA default) only when `SPTC_CYCLIC_PMAP` is unset.
**Fix:** cite `ta_builder.h:430-431` in all four places, and reword: the pmap slot is wired
(defaults to TA's blocked pmap unless `SPTC_CYCLIC_PMAP` is set) — the §8 lever is plumbed, not absent.

### A3. `sptc_traced_einsum.h` path + env var wrong (MPQC_COMPARISON.md:128, 259)
Claim: `src/sptc_traced_einsum.h`, env `SPTC_TRACE_OPS`. Reality: file is **`tools/sptc_traced_einsum.h`**;
the only env var is **`SPTC_TRACE_OPS_PATH`** (output path, :36); the trace is enabled at *compile
time* (compiling the traced generated TUs against the header), not by an `SPTC_TRACE_OPS` env knob
(which does not exist).
**Fix:** `src/`→`tools/` at both lines; correct the env-var name; drop the "env-gated knob" framing at :259.

---

## B. UNSUPPORTED (no committed backing)

### B1. `~87% of cold T2` / `~100× off peak` — no committed profile (MPQC_EVALUATION.md:200-201, 356, 368, 426; CONTRACTION_IR.md:19, 104; MPQC_COMPARISON.md:404, 440)
No `steps.csv` / per-op timing artifact is committed (`docs/trace/` holds only instrumented
*source*; no CSV has an `87`/off-peak column). The cross-ref at EVAL:200 to "MPQC_COMPARISON.md §3"
is also stale — §3 (lines 73-102) states "~43-50% of size-1 T2 wall was MADNESS task-scheduling"
on the old size-1 config, a *different* quantity. The figure appears only as recurring prose,
attributed at EVAL:368 to "campaign notes."
**Fix:** commit the per-op trace/`steps.csv` that yields the 87%/100× split and cite it, or mark
the figure "(from campaign per-op trace, not committed here)"; fix the stale §3 cross-ref (done — EVAL:200).
**Investigated (2026-07-30):** the tooling exists but doesn't produce this cleanly out of the box —
`build-cd53bd3-clang/ta_traced_ops` (arena) segfaults on C4H10's giant intermediate after ~21 ops;
the owning-ToT `ta_traced_main` target (build-auxbatch) has object files but no linked binary. Backing
the figure = build+link the owning traced binary, run on a cc-pVTZ molecule, and take the max-op wall
share. Note the header caveat: per-op times are *fenced/serialized*, so the share is a serial-op-work
fraction, and "~100× off peak" (a FLOP-efficiency metric) needs FLOP counts the trace doesn't emit.

### B2. `proto=45 11.6 s` np1 — not in any CSV, conflicts with the grid (MPQC_COMPARISON.md:368)
Claim: single-node C₂H₆ cold proto=45 11.6 s vs proto=100 18.9 s; at np=16, 6.6 vs 13.8 s.
**Investigated (2026-07-30):** the **np16 pair is backed** — proto45 6.6 s (grid) and proto100
13.8 s (`proto100_cold.csv` C2H6 np16 = 13.758). The **np1 pair (11.6 / 18.9) is not** — the grid
gives proto45 np1 = 12.4 s (a ~6.5% conflict with 11.6), and `comparison.csv` np1 has proto45 = 80.0
(arena, giant intermediate) / proto100 = 27.0 — neither is 11.6 or 18.9. They appear to be from an
ad-hoc owning-ToT np1 micro-benchmark not saved to a CSV.
**Fix:** reconcile the np1 proto=45 number to the grid (12.4 s) or commit the micro-benchmark that
produced 11.6/18.9; the np16 pair is fine as-is.

---

## C. DRIFT / OVERSTATED (self-inconsistent or stale after the partial refresh)

### C1. EVAL lever (b) still says "port aux-Κ batching" (MPQC_EVALUATION.md:433-436) — the partial-refresh straggler
Stage 10 (:238) says aux-Κ batching is "implemented and validated (2026-07-30)"; MPQC_COMPARISON.md:447
says "DONE (memory). Now implemented." But lever (b) still reads "Hexane memory wall → **port**
aux-Κ batching … a generator/backend project. This is the memory fix." Same doc contradicts itself.
**Fix:** rewrite lever (b) to past tense/DONE, matching Stage 10 and CMP:447.

### C2. "pmap co-location is THE actionable cold-gap lever" (EVAL:28, 35-37) contradicts the settled §8
The intro presents pmap/tiling as the actionable lever; the body concludes the opposite —
EVAL:369-370 "lever (a) as 'fix the input pmap' is … refuted," EVAL:430 "not an in-repo lever …
needs a TA backend fix," and README:22-23 / MPQC_COMPARISON.md both say the lever is the ToT-einsum
representation, "not tiling or pmap — both empirically ruled out" (cyclic-pmap = no speedup:
`cyclic_pmap_timing.csv` default median 48.85 s vs cyclic 49.90 s, checksum-invariant).
**Fix:** revise EVAL:24-37 so the intro marks pmap/tiling as the refuted initial hypothesis and
names the ToT-einsum per-outer-cell overhead (TA-backend/derivation fix) as the real lever.

### C3. "byte-identical math" overstated (MPQC_EVALUATION.md:244; MPQC_COMPARISON.md:401)
Batched vs unbatched is called "byte-identical math." `auxbatch_correctness.csv` shows the T2 sums
differ by float reassociation (C₂H₆ agree to 10 sig figs, differ at the 11th; C₄H₁₀ at ~13th) — and
the doc itself says "match to 10-13 significant figures" two paragraphs later.
**Fix:** "algebraically identical (same cell count, same flops); results agree to 10-13 sig figs
modulo Κ-sum reassociation" — not "byte-identical," which reads as bit-identical output.

### C4. Headline "~4.5–7× slower cold at np=16" overstated (MPQC_COMPARISON.md:30)
np=16 cold ratios (raw CSVs) are C₃H₈ 4.42×, C₄H₁₀ 4.76×, C₅H₁₂ 5.63× — max ~5.6×. The 7× cell is
np=**4** (C₄H₁₀ 99.3/13.9 = 7.1×).
**Fix:** either drop "at np=16" (line 326 states the same 4.5–7× range without the qualifier, which
is defensible across np4–16) or change to ~4.5–5.5×.

### C5. `SUMMARY.md` "Cold gap (~10× np1)" doesn't match its own cold table (SUMMARY.md:33)
The "Repro COLD T2" table (from `results.csv`) gives C₂H₆ np1 = 12.2 s vs MPQC cold 7.7 s = **1.6×**
(C₃H₈ 2.1×, C₄H₁₀ 2.0×). The ~10× comes from a *different* cold dataset in `comparison.csv` (repro
cold 80–81.8 s / MPQC 7.732 = 10.3–10.6×; the proto45-shipped TPD=8 DF-half-transform config). Two
"cold" magnitudes differing ~6.5× are both labeled "cold" in one doc.
**Fix:** label the two cold datasets distinctly; clarify ~10× is the shipped DF-half-transform
config, not the 12.2 s multi-rank-table np1.

### C6. `SUMMARY.md` warm "C2H6 1.3×" — inverted direction + not series-wide (SUMMARY.md:32)
The warm table's `ratio(np1)` column is repro/MPQC (C₂H₆ 0.77× = repro faster); the "1.3×" in
Findings is the reciprocal MPQC/repro, with no note. Also "repro ~ MPQC warm" holds only for the two
smallest molecules — the ratio grows 0.77× (C₂H₆) → 1.26× (C₃H₈) → 1.69× (C₄H₁₀) → 2.93× (C₅H₁₂).
**Fix:** state "C₂H₆ 0.77× (repro faster)" to match the table, or note 1.3× is MPQC/repro; scope the
parity claim to small molecules / give the range.

### C7. EVAL "warm-vs-warm ~1.3× (§6/§11)" — cherry-picked + wrong section (MPQC_EVALUATION.md:154)
~1.3× matches only C₃H₈ cc-pVTZ warm np1; §11's headline is "repro ~2–3× slower warm"; the §6
citation is wrong (§6 gives cold-to-cold 1.44× faster, not warm 1.3×; §7 gives warm ethane 3.3×
faster).
**Fix:** name the molecule/basis/footing (C₃H₈ cc-pVTZ warm np1), reconcile with §11's "~2–3× slower
warm," and fix the §6 citation.

### C8. proto=100 divergence: ~0.2% (CMP) vs ~7% (SUMMARY/comparison.csv) (SUMMARY.md:34, comparison.csv:6 vs MPQC_COMPARISON.md:372)
Same quantity (proto=100 cc-pVTZ t-dependent divergence) reported as ~0.2% in CMP and ~7% in the two
data files — a ~35× unreconciled gap. CMP frames ~0.2% as the refined value.
**Investigated (2026-07-30):** the *raw* proto100-vs-proto45 T2 sum diverges far more than either
figure — `proto100_cold.csv` C2H6 np16 t2_sum = −0.001426 vs proto45 −0.000677 (>100%); C3H8
proto100 = +28.97 vs proto45 −0.044 (sign flip). So **neither 0.2% nor 7% is the raw-sum metric** —
they must measure a different, gauge-stable quantity (energy? a specific amplitude norm?).
**Fix:** state explicitly what quantity the 0.2%/7% measure, then reconcile the two figures; as
written they are ambiguous *and* inconsistent with the raw residual divergence.

### C9. t-indep counts 55/252 (CMP) vs 54/198 (IR) not cross-noted (MPQC_COMPARISON.md:173-175, CONTRACTION_IR.md:112-113)
CMP counts flat straight-line statements (252 total, 55 t-indep); CTIR collapses SSA-versioned slots
into 198 values (54 t-indep, 33 persistent). Both correct, but a reader sees 55-vs-54 / 252-vs-198.
**Fix:** add a one-line reconciliation noting the two count different objects.

---

## D. STALE-CITATION (line-drifted or wrong file/function)

| # | Doc location | Claim | Correct target |
|---|---|---|---|
| D1 | MPQC_EVALUATION.md:43-44 | `evaluate_csv_closedshell` builds the residual set at cck.ipp:1513-1523 | `1513-1523` belongs to **`generate_csv_closedshell`** (:1507); `evaluate_csv_closedshell` (:1584) only adds the "once" guard at :1591 |
| D2 | MPQC_EVALUATION.md:126-127 | make_R_template_csv annotation at cck.ipp:895 | :895 is the non-CSV spin-traced path; correct is **cck.ipp:1706-1708** — drop :895 |
| D3 | MPQC_EVALUATION.md:207-209 | cd53bd3 assertion-drop "exercised at cck.ipp:1619" | :1619 is a comment fragment; the batched-SUMMA install is **cck.ipp:1643** (block 1601-1666) |
| D4 | MPQC_EVALUATION.md:222 | set_custom_evaluator at cck.ipp:1644 | call token at **cck.ipp:1643** (spans 1643-1645) |
| D5 | MPQC_EVALUATION.md:196 | env knobs + outer TiledRange at ta_builder.h:314-323 | knobs at **:60-62, :293-298**; TiledRange build at **:380-390** (:314-323 is Pass-0 grouping) |
| D6 | MPQC_EVALUATION.md:290 | pair-key dims forced to size 1 at ta_builder.h:319-321 | size-1 forcing at **ta_builder.h:386-389** (:388) |
| D7 | MPQC_COMPARISON.md:48 | einsum backend `result.hpp:379,613-628` | einsum call at **:381** (:379 is `ArrayT result;`); qualify path `backends/tiledarray/result.hpp` (core/eval/result.hpp is only 496 lines) |
| D8 | MPQC_COMPARISON.md:137 | g·g intermediate at generated_t2:211 | **generated_t2_residual.cpp:212** (:211 is a `// release`) |
| D9 | MPQC_COMPARISON.md:138 | g·C giant at generated_t2:486 | **generated_t2_residual.cpp:487** (:486 is a `// release`) — matches the other cites of this einsum |
| D10 | CONTRACTION_IR.md:98 | worked-example CSE37 `uses=1` | `.ctir` source says **`uses=2`** (whole_t2_residual.ctir:864; CSE37 consumed at :220 and :869) |
| D11 | MPQC_COMPARISON.md:288 | intro C₄H₁₀→10.2 s (7.6×), C₅H₁₂→20.4 s | fresh grid uses 10.3 s (→7.5×) / 20.5 s; intro cites the stale committed CSV, grid cites fresh /proj |
| D12 | SUMMARY.md:28 | C₄H₁₀ cold np16 = 48.4 s | superseded by fresh grid **48.9 s** (results.csv Jul-28 vs the 2026-07-30 re-sweep) |

---

## E. Re-measurement (this node + node16–31 re-sweep)

### E1. README checksum table — CONFIRMED correct (do NOT edit)
`build-owning` + the README recipe reproduces all 8 values exactly: T1 nnz 522, sum −0.0888799157,
sumsq 0.0026454022, max_abs 0.0151662826; T2 nnz 98598, sum 0.2141888066, sumsq 0.0997316723,
max_abs 0.0174497557. (The table is also independently backed by `mpqc-benchmark/traces/checksum-run`.)

### E2. §11 grid IS backed — but by uncommitted data (fix = CSV sync, not a number change)
The doc §11 grid matches the fresh `/proj/.../jianjian-scaling/results.csv` (Jul-30 06:37) exactly
(C₄H₁₀ cold np16 = 48.9357, C₅H₁₂ cold np1 = 391.524); the **committed** `results.csv` is the stale
Jul-28 version (48.415; C₅H₁₂ np1 absent). nnz/checksums are rank-invariant and identical across
versions. An independent node16–31 re-sweep (into `jianjian-scaling-verify/`) is confirming
reproducibility a third time.
**Fix:** sync the fresh `/proj` `results.csv`, `results_warm.csv`, `mpqc_mr.csv` into
`docs/scaling-campaign-data/`.

### E3. MAJOR — repro hexane is NO LONGER blocked (the doc's Hexane conclusion is going stale live)
The docs (MPQC_COMPARISON.md §11 "Hexane"; EVAL) say repro hexane is "blocked by a truncated DF leaf
… aborts with heap corruption in the T1 residual." As this report is written, a repro hexane run on
**regenerated complete leaves** (`leaves/C6H14_fixed` → `C6H14_g0coo`, g `shape=376,376,906
nnz=127481856`, symmetric) has **loaded fine (213.9 s) and completed T1 cleanly** (nnz=1785,
sum=−0.00329269345, max_abs=0.00269084, wall 53.8 s — no heap corruption), with T2 running batched at
~25 GB. So with complete leaves + aux-batching, repro hexane runs; the "aborts in T1 on the truncated
leaf" text describes the *old* truncated leaf, which has since been regenerated.
**RESOLVED (T2 completed, rc=0):** repro hexane completes single-rank — **T1** nnz 1785 /
sum −0.0032926934509 / 53.8 s / 13.5 GB; **T2** nnz 591501 / sum −10.684264217531 / sumsq
39.7679866480 / max_abs 0.60932854 / 933.4 s / **peak 27.5 GB** (unbatched would need ~130 GB).
Data: `scaling-campaign-data/repro_hexane_batch.csv`. The author independently updated
MPQC_COMPARISON.md §11 to reflect this during the run. *(The workflow's adversarial pass correctly
refuted this as a drift **against committed data** — the committed doc said "awaits clean leaves,"
true then; the live measurement supersedes it.)* **Remaining check:** repro hexane T2 correctness
vs MPQC needs an MPQC hexane residual checksum (mpqc_hexane_batch.csv has timings only) — see E4.
**Dedup:** two files hold this result — `repro_hexane.csv` (author) and `repro_hexane_batch.csv`
(this pass, doc-referenced); keep one.

### E4. Flag — MPQC hexane energy discrepancy (reconcile)
`mpqc_hexane_batch.csv` reports energy −235.44957 (perf-batch, aux_target_size=128). A separate np16
variant logged energy −236.58368 (`mpqc_C6H14_g0dump_np16.log`, aux_target_size=32) — ~1.13 Hartree
apart. Batching should be transparent, so these are likely different configs/truncations; the doc's
−235.44957 needs an independent perf-batch re-run to confirm. *(Deferred: single-rank uncapped hexane
dump OOMs node3; the perf-batch timing re-run is queued behind the live hexane T2.)*

### E6. Independent re-sweep (node16–31) — §11 grid reproduces (Phase 3)
A fully independent re-sweep (fresh output dir, both sides) reproduces the committed §11 grid
within run-to-run noise, with **bit-stable checksums**:
- **Cold T2:** C2H6 −0.7%/+1.6%, C3H8 +0.5%/−1.6%, C4H10 +0.5%/−2.3%, C5H12 np16 +3.8% (max).
  C4H10 np16 `t2_sum = −7.10587645551`, nnz 371400 — identical to the committed value.
- **Warm T2:** C2H6 −0.0%/−2.3%, C5H12 np16 −2.0%.
- **MPQC multi-rank cold:** C4H10 np16 10.31 → 10.15 s (−1.5%).
- A few of the longest cells (C5H12 np1 warm, C5H12 np16 MPQC) didn't finish in the re-sweep window,
  but their committed values were already validated by the fresh sweep the grid was built from.
No committed number failed to reproduce. **Verdict: the §11 grid is sound.**

### E5. aux-batching A/B — spot-confirmed
C₂H₆ ats0 reproduced (peak 3.46 GB, T2 checksum matches committed). The committed RSS table
(−43/−58/−63/−53%) is arithmetically correct against `auxbatch_correctness.csv`.

---

## F. Candidate findings REFUTED on adversarial re-check (documented, NOT acted on)

- **occ17_t4 "~0.03%"** (EVAL:325): 0.02534% rounds to 0.03% with the explicit tilde — legitimate.
- **README checksum "no CSV backing"**: the table is backed by `mpqc-benchmark/traces/checksum-run/
  sptc_coo_iter1` (the source the README names), not the scaling CSVs — comparing them was
  apples-to-oranges.
- **Hexane memory-wall "drift"**: against *committed* data the layering is chronological and
  self-consistent (unbatched OOMs = true; "stays out regardless" attributed to heap corruption /
  "awaits clean leaves"). Superseded only by the live E3 measurement, not by anything in the repo.

## G. Verified accurate (spot list — no change)
- Same TiledArray `cd53bd3` both sides: repro builds `third_party/tiledarray-cd53bd3-clang`
  (setup-tiledarray.sh + CMakeLists), MPQC tracks cd53bd3. The sequant-fork's different TA pin
  (`b8c1d75`) only affects codegen, not the runtime comparison.
- Docs do **not** cite a nonexistent `make_custom_evaluator` (only `make_batched_custom_evaluator`).
- cyclic-pmap non-effect, occ-tile-size non-lever, coarse-occ numerical hazard: all backed by their CSVs.
- Two distinct segfaults (driver wait_policy vs np≥8 ArenaTensor race) are correctly kept separate.
- aux-batching "memory not speed" (ratios unchanged): A/B wall times ats0≈ats96 confirm.
