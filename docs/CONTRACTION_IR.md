# CTIR — a human-readable Contraction IR for the CSV-CCSD residual

*Why the generated `TA::einsum` C++ is not enough to describe the computation, and a small
intermediate representation that is. Emitter: `ContractionIRGenerator`
(`sequant-fork/SeQuant/core/export/contraction_ir_generator.hpp`), a `Generator<Context>` backend
wired into the derivation driver next to the einsum export. Committed samples:
`docs/scaling-campaign-data/whole_t{1,2}_residual.ctir`.*

## Why the einsum isn't enough

`src/generated_t2_residual.cpp` is a faithful *execution artifact* — a flattened, linear list of
`TA::einsum` calls — and precisely because of that it is a poor description of **what is being
computed** and of **what MPQC does**. Five concrete gaps:

1. **It's flattened, not a DAG.** The contraction tree is gone; sharing survives only as a
   variable reused by several later statements (`CSE37` appears in N places). You cannot see the
   computation's structure, or *why* an intermediate is shared, or how much.
2. **No cost/size — so the dominant cost is invisible.** There are no cell counts, inner extents,
   or per-cell figures. The contraction that is ~87 % of cold T2 and runs ~100× off peak
   (`generated_t2_residual.cpp:487`) looks *identical* to a cheap one. This whole repo's cold-gap
   investigation had to reverse-engineer, by profiling and by 16-node experiments, a fact the IR
   states at emit time: that intermediate is **1.6 M tiny per-pair ToT cells**.
3. **No domain semantics.** The `;` in `"i_1,i_2,μ̃,Κ;a"` splits ToT outer/inner but never says
   the inner axis `a` is a *per-pair PNO domain* keyed by the occupied pair `(i,j)`, nor that μ̃
   (PAO) and Κ (DF aux) are global (non-proto). That proto structure is exactly what forces μ̃,Κ
   to be block-sparse *outer* axes (and, per `MPQC_EVALUATION.md` §8, what makes the tiny-cell
   half-transform unavoidable).
4. **It cannot express what MPQC does at all.** MPQC's real computation is a runtime-scheduled DAG
   over a `CacheManager`: amplitude-*independent* (`t-indep`) intermediates that are **built once
   and reused across CC iterations** (persistent), amplitude-dependent (`t-dep`) ones rebuilt every
   iteration, canonical-key cross-term sharing gated by `min_repeats=2`, and optional aux-Κ
   *batched* evaluation. A static einsum call list has **no surface** for any of these — so it
   literally cannot describe MPQC's computation, only one fixed lowering of one term set.
5. **Manual `// release` ≠ cache lifetime.** The `X = ArrayToT(); // release` lines are the
   codegen's own ref-count liveness; they are easily mistaken for MPQC's `reset()`/persistence,
   which encode a different thing (per-iteration vs cross-iteration survival).

The `EvalExpr`/`EvalNode` tree and the per-term `r2_terms.tsv` each fix *some* of this but not all:
the tree is per-term (no cross-term DAG) and carries no runtime semantics; the TSV is per-term,
un-shared, and cost-free. CTIR is the missing view: one shared DAG with cost and runtime semantics.

## What CTIR shows

CTIR renders the whole-residual computation as a **value DAG**: the export framework hands the
emitter an imperative stream that *reuses* C++ slot names across live ranges, and CTIR recovers one
`def` per computed **value** (a slot reused after a `free` becomes a new versioned value `name#2`;
an accumulated value — the residual, or an intermediate built by `+=` — stays *one* value rendered
as `= Σ`). Cross-term sharing then shows directly as `uses=N`. Grammar (informal):

```
spaces:  i=<n>  μ̃=<n>  Κ=<n>  a=PNO⟨per-pair⟩~<n>
leaves:
  <name>  [<outer> ; <inner>]⟨<pair-key>⟩  {tot|flat}  {t-indep | t-dep(amplitude)}
computation:
  {def|result} <value-id> [<outer> ; <inner>]⟨<pair-key>⟩ {tot|flat}  uses=N  {t-indep|t-dep}  [persistent: …]
      = [<scalar>] {contract{<summed idxs>} | copy} <operand> * <operand>      (single contribution)
      = Σ <k> contributions:                                                    (accumulated value)
          <scalar> contract{…} <op> * <op>
          …
      cost: cells=<#outer ToT cells>  inner=<per-pair extent>  flops=<…>  per-cell=<…>
            [⚠ CELL-BOUND (<n> tiny ToT tasks)]  [· batchable/Κ (aux)]
    free <value-id>        ; repro: release slot after last use (static liveness)
summary:  <#values> (t-indep/t-dep)  ·  cold-precompute flops share  ·  #persistent  ·  #batchable
```

Index names are canonicalized per value (SeQuant's internal numeric tags stripped; same-space
indices disambiguated — occ as `i j k …`, others primed as `μ̃ μ̃'`). Every annotation is derived
from the real code, not guessed:

| annotation | meaning | source |
|---|---|---|
| `[outer ; inner]⟨pair-key⟩` | ToT block-sparse outer vs per-pair PNO inner; `⟨…⟩` = the occupied pair keying the PNO domain | `classify_indices`/`tot_indices` (`utility/indices.hpp`) |
| `uses=N` | # consumers of this **value** on the DAG; `N≥2` = a cross-term shared node (what MPQC caches with `min_repeats=2`) | whole-DAG operand count |
| `cells` / `⚠ CELL-BOUND` | # outer ToT cells = # independent per-cell tasks; flagged for the dominant-cell-count ToT node(s) (relative, so it travels across problem sizes) | product of outer + pair-key extents (via `idx_to_extent`) |
| `t-indep` / `t-dep` | amplitude-independent (build-once candidate — the "cold precompute") vs amplitude-dependent (rebuilt each iteration — the "warm" work). Resolved by a **whole-DAG volatility fixpoint**: t-dep iff it or any operand transitively contracts a `t` leaf | `label == "t"` volatility, matching MPQC's `is_volatile` predicate (`cck.ipp:1640`) |
| `[persistent: built once, reused across iters]` | a `t-indep` value consumed by a `t-dep` value — MPQC's `CacheManager` builds it once and it survives `reset()` across CC iterations | `cache_manager.hpp` rule (V→NP, NV-with-V-consumer→P) |
| `· batchable/Κ (aux)` | the contraction sums over the DF aux index, so it can be streamed in Κ-slices to bound memory | the aux-Κ batching hook (`make_batched_custom_evaluator`, `cck.ipp:1601-1645`) |
| `free …` | the repro's *static* release point (whole-DAG last-use) — contrast with MPQC's cross-iteration cache lifetime above | export ref-count liveness |

## Worked example — the giant DF half-transform

The one contraction that dominates cold T2. **As einsum** (`generated_t2_residual.cpp:487`) it is
indistinguishable from any other line:

```cpp
I_ap2_μ̃_Κ("i_1,i_2,μ̃_19906,Κ_1;a_1") =
    TA::einsum(g_μ̃_μ̃_Κ("μ̃_19905,μ̃_19906,Κ_1"),
               C_ap2_μ̃("i_1,i_2,μ̃_19905;a_1"), "i_1,i_2,μ̃_19906,Κ_1;a_1");
```

**As CTIR** (`docs/scaling-campaign-data/whole_t2_residual.ctir`) it announces exactly the facts
this repo had to discover the hard way:

```
def I_ap2_μ̃_Κ [μ̃ Κ ; a]⟨i j⟩ tot  uses=1  t-indep
    = contract{μ̃'} g_μ̃_μ̃_Κ * C_ap2_μ̃
    cost: cells=1.6e+06  inner=45  flops=8.1e+09  per-cell=5130   ⚠ CELL-BOUND (1.6e+06 tiny ToT tasks)
def CSE37_i_i_ap2_ap2_Κ [i j Κ ; a a'] tot  uses=1  t-indep  [persistent: built once, reused across iters]
    = contract{μ̃} I_ap2_μ̃_Κ * C_μ̃_ap2
```

Read off directly: it is **t-indep** (amplitude-independent → MPQC builds it once, not every
iteration; the repro's cold driver rebuilds it every pass — the warm/cold gap in one word); it is
**CELL-BOUND** with **1.6 M** tiny per-pair ToT cells (the ~100×-off-peak tiled-task overhead that
`MPQC_EVALUATION.md` §8 identified as the real cold bottleneck — occ tiling and pmap were both
empirically ruled out); and it carries μ̃ and Κ as *outer* block-sparse axes with only the small
`a` PNO domain inner (the proto structure that makes the tiny cells unavoidable, and that no
generator knob can reshape — `MPQC_EVALUATION.md` §8). Its consumer `CSE37…` is the `t-indep`/
`t-dep` boundary, so CTIR marks *it* `[persistent: built once, reused across iters]` — the exact
`CacheManager` entry MPQC keeps alive across CC iterations. None of this is visible in the einsum.

The whole-residual `summary:` block makes the split quantitative: for T2, **54 of 198 values are
t-indep (~71 % of total flops), 33 of them persistent** — i.e. most of the residual's arithmetic is
amplitude-independent work MPQC builds once and the repro's cold driver rebuilds every pass, plus
**37 aux-Κ-batchable** contractions. That is the warm/cold and hexane-memory story (`MPQC_EVALUATION.md`
§6/§10) in three lines the einsum cannot express.

Cross-term sharing is legible too: e.g. `def CSE6_i_i_i_ap2 … uses=24` and `CSE4_… uses=22` show
single intermediates feeding 20+ downstream contractions — the reuse MPQC's runtime cache exploits
and that the einsum only expresses as an opaque repeated variable.

## Producing it

The emitter is a `Generator<Context>` subclass, modeled on `TextGenerator`, that collects the DAG
during the framework's `declare`/`compute`/`unload` callbacks and renders at `get_generated_code()`
(two passes are needed: `uses` counts and the persistent boundary are whole-DAG properties). It is
wired into `sequant-fork/tests/manual/test_csv_ccsd_derivation.cpp` right before the einsum export,
on a copy of the same forest, writing `generated_R{1,2}.ctir`. Regenerate with the
`ta_generator_cc_test` target (see that repo); the einsum output and the shipped
`generated_t2_residual.cpp` are untouched (the CTIR emit is additive).

CTIR is deliberately **descriptive, not a lowering IR** — its job is legibility and making the
repro(static)-vs-MPQC(runtime) difference explicit, which is exactly the gap the einsum leaves.
