# Layered IRs: lowering the CSV-CCSD residual from the MPQC equation to einsum sequences

*A design + feasibility study. Question: can the closed-shell CSV-CCSD residual be expressed as a
stack of **layered intermediate representations** that progressively lower — MLIR-style — from the
high-level MPQC/SeQuant tensor equation down to a concrete sequence of `TA::einsum` calls, with
each level independently inspectable and each boundary an explicit pass? Short answer: **yes**, and
most of the stack already exists as real data structures — the pipeline is already an informal
progressive lowering. This document formalizes the layers, names the passes, localizes the
MPQC-vs-repro difference to a single pass, and scopes what it would take to make the stack real.
Companion to `MPQC_EVALUATION.md` (what MPQC does) and `CONTRACTION_IR.md` (the CTIR value-DAG).*

---

## 1. The pipeline is already an informal progressive lowering

Tracing the real code end to end (SeQuant fork + MPQC `cck.ipp`), the residual passes through a
sequence of representations — but implemented as **one derivation → one binarization → ~8 parallel
export backends**, with several level boundaries *fused or implicit* rather than explicit:

| # | representation | concrete type | boundary quality |
|---|---|---|---|
| 1 | high-level tensor equation | `ExprPtr` (`Sum` of `Product` of `Tensor`) | **clean type** |
| 2 | DF- + CSV-decorated equation | *same* `ExprPtr` (mutated in place) | **fused** (no distinct type) |
| 3 | factorized + reordered equation | *same* `ExprPtr` | **fused** (tree built then discarded) |
| 4 | binary contraction forest | `EvalNode<EvalExpr>` = `FullBinaryNode` per summand | **clean type** (the pivot) |
| 5 | scheduling (cache P/NP, aux-Κ batching) | — *runtime overlay, no type* | **implicit** (MPQC path only) |
| 6 | tiling / distribution | `TiledRange`, `SparseShape`, `Pmap`, `ArrayToT` | **orthogonal** (TA-side) |
| 7 | concrete einsum sequence | C++ text via `Generator<Context>` | **clean plug-in** |

Three facts from the trace shape everything below:

- **Two clean typed layers already exist and are load-bearing.** The symbolic `ExprPtr`
  (`make_cceqvec_csv_closedshell`, `mpqc4/.../cc/sequant.cpp:156`) and the binary forest
  `EvalNode<EvalExpr>` (`sequant-fork/.../eval/eval_expr.hpp:395`, `using EvalNode =
  FullBinaryNode<T>`). The forest is the pivot that the *entire* export framework consumes.
- **The emission boundary is already a clean plug-in.** `Generator<Context>`
  (`SeQuant/core/export/generator.hpp:40`) + the `export.hpp` driver
  (`export_group`→`GenerationVisitor`) is implemented by ~8 backends today: `TiledArrayGenerator`,
  `ItfGenerator`, NumPy/PyTorch einsum, Julia (TensorKit/ITensor), `TextGenerator`, and — most
  relevant here — `ContractionIRGenerator` (CTIR).
- **There is no IR→IR lowering.** No backend consumes another backend's output; scheduling
  (`CacheManager` persistence + aux-Κ `make_batched_custom_evaluator`) is a *runtime overlay on the
  forest*, present only in MPQC's evaluation path (`cck.ipp:1601-1645`, `eval.hpp:1129`) and with
  **no surface at all** in the flat einsum artifact. This missing middle is exactly what a layered
  design would add.

---

## 2. The proposed layered stack

Five levels, high to low. Each names its abstraction, the concrete type to reuse, and the lowering
pass to the next level (with where that pass already lives). *Exists* / *new* marks what is real
today vs. what a layered implementation would add.

### L4 — Tensor-Equation IR  *(exists)*
Spin-free, biorthogonally-transformed CCSD residual as an `ExprPtr` `Sum` of `Product`s of
abstract `Tensor`s over occ/virt spaces. **No DF, no CSV, no PNO domains, no tiling.**
Produced by `make_cceqvec_csv_closedshell` (`sequant.cpp:156-211`): `CC{k}.t()` + per-rank
`tail_factor` + `biorthogonal_transform` + `simplify`, under `SPBasis::Spinfree` and
`mbpt::Context{.csv=CSV::Yes}`. Textual form: SeQuant's serialization (the `{bra;ket;aux}`
notation already used in `r2_terms.tsv`).

> **↓ Pass P4 — DF + CSV projection.** `density_fit` replaces each 4-index `g` with two 3-center
> factors carrying a fresh **aux index Κ** (`df.cpp:18-99`); `csv_transform` mints **PAO indices μ̃**
> and inserts per-pair **PNO coefficient tensors `C`** linking `a<i,j>` to μ̃ (`csv.cpp:23-144`);
> then `flatten`. Call site `cck.ipp:1520-1523`. *Today: mutates the same `ExprPtr` in place — the
> DF-factorized and CSV-expanded forms are just successive snapshots of one object.*

### L3 — Domain-Typed Tensor IR  *(exists as a snapshot; type not distinct)*
`ExprPtr` where indices now carry **space + per-pair proto decoration** (`a<i,j>` = PNO restricted
to occupied pair). The tensor-of-tensor **outer/inner structure is entirely implicit in this
decoration**: non-proto axes (`i`, `μ̃`, `Κ`) are the block-sparse *outer* modes, proto axes are
the per-pair *inner* mode. It never becomes a distinct type — it materializes only as the einsum
`";"` (L0) or the `ArrayToT` cell structure (tiling). *A layered design would make the outer/inner
split a first-class node attribute here rather than something re-derived at every backend
(`tot_indices`, `classify_indices`).*

> **↓ Pass P3 — factorize + binarize.** `optimize(OptFor::Flops, idx_to_extent, is_volatile_leaf=
> label==t, n_replay, ReorderSum::Reorder)` picks a pairwise contraction order per summand via a
> flop cost model (`optimize.cpp:35-169`, `single_term.hpp:368`) and reorders summands to cluster
> shared intermediates; then `binarize` builds the tree. *Today `optimize()` builds a binary tree
> internally just to score the reorder and **throws it away** (`optimize.cpp:155`), returning an
> `ExprPtr`; `binarize` re-derives it at L2. A layered design would keep the tree as the artifact.*

### L2 — Contraction-Forest IR  *(exists — the pivot)*
`EvalNode<EvalExpr>` — a **forest of per-summand binary trees**. Each node carries `op_type`
(`Product`/`Sum`/`Adjoint`), `result_type`, canonical index order, a CSE `hash_value` (equal for
results equal modulo phase), and `tot()` (has-proto ⇒ ToT). This is the portable, backend-agnostic
contraction IR and the natural place to **anchor the stack**: everything above is symbolic algebra;
everything below is scheduling and resources. In the export path the node is `ExportNode<ExportExpr>`
(`ExportExpr : EvalExpr` adds a stable `id()`).

> **↓ Pass P2 — schedule (THE pass that matters, and the one that's split/implicit today).**
> Turns the abstract forest into a concrete execution plan by making four decisions:
> 1. **cross-term CSE** → a shared value-DAG (`eliminate_common_subexpressions`, or the runtime
>    cache's canonical-key collisions);
> 2. **cache/persistence** → classify each value persistent (amplitude-independent, build-once,
>    survives `reset()` across CC iterations) vs volatile (`cache_manager.hpp`, `min_repeats=2`);
> 3. **tiling + pmap** → occ tile size, `TiledRange`, process map;
> 4. **aux-Κ batching** → stream Κ in tile-aligned slices, sum partials, scale the sparse
>    threshold by `1/n_batches` (`make_batched_custom_evaluator`, `eval.hpp:1129`).
> **This is where MPQC and the repro diverge** (§4). Today it has no IR: MPQC realizes it as a
> *runtime overlay* (`CacheManager` + custom evaluator + a pmap inherited from the CSV energies
> array, `cck.ipp:1567-1645`); the repro realizes it *statically and piecewise* (generation-time
> CSE + `build_tot_array` tiling + a hand-written `aux_k_batching.h` transform of the flat einsum).

### L1 — Scheduled / Resource IR  *(the one genuinely new artifact; CTIR is its prototype)*
A value-DAG in which the P2 decisions are **first-class**: shared intermediates (with use counts),
cache lifetime (persistent/volatile), tiling (occ tile / trange), pmap, and aux-Κ batch factor.
This is a concrete, backend-agnostic **execution plan**.

**CTIR (`CONTRACTION_IR.md`, `contraction_ir_generator.hpp`) already is a *descriptive* prototype
of this level** — it renders exactly this value-DAG and annotates `t-indep`/`t-dep` (= cache class),
`cells` + `⚠ CELL-BOUND` (= tiling cost), `[persistent]` (= cross-iteration reuse), and
`batchable/Κ`. The difference between CTIR and a real L1 is: CTIR *describes* these as read-only
annotations; an executable L1 would carry them as **decisions that lower to L0** (choose the tiling,
choose which values are batched, choose the pmap) and be the input the emitter consumes.

> **↓ Pass P1 — emit.** Walk the scheduled DAG and produce the target: `TA::einsum` /
> `einsum<DeNest::True>` calls via `Generator<Context>`/`export.hpp`, plus the tiling realized by
> `build_tot_array` and the batching realized by `aux_k_batching.h`. *Today the emitter (P1) reads
> L2 directly and the scheduling is baked into the runtime or hand-written; a layered design routes
> it through L1.*

### L0 — einsum sequence  *(exists — the target)*
The flat `TA::einsum` C++ (`src/generated_t{1,2}_residual.cpp`) run against `build_tot_array`-tiled
`ArrayToD`/`ArrayToT` tensors, optionally wrapped by aux-Κ batching. Equivalently, MPQC's runtime
interpreter emits the *same* primitive calls node by node.

---

## 3. One term, all the way down

The giant DF half-transform — the dominant cold-T2 block (its ToT×ToT consumer `generated_t2:488` is
≈40 % of the cold einsum-region per `gap_profile.txt`; the earlier "~87 %/~100×-off-peak" figure is
superseded — see `MPQC_PROFILE_DEEP.md`) — at each level:

```
L4  Tensor-Equation      … g{i j; a b} · t{a b; i j} …            (abstract 4-index ERI × amplitude; no DF/CSV)
L3  Domain-Typed         g_μ̃μ̃Κ[μ̃',μ̃,Κ] · C_ap2_μ̃[i,j,μ̃'; a<i,j>] …   (DF aux Κ; PAO μ̃; per-pair PNO a<i,j>)
L2  Contraction-Forest   Product node:  result [μ̃ Κ ; a]⟨i j⟩  = einsum(g_μ̃μ̃Κ, C_ap2_μ̃) over μ̃'
L1  Scheduled  (= CTIR)  def I_ap2_μ̃_Κ [μ̃ Κ ; a]⟨i j⟩ tot uses=1 t-indep
                             = contract{μ̃'} g_μ̃_μ̃_Κ * C_ap2_μ̃
                             cost cells=1.6e6 inner=45 … ⚠ CELL-BOUND · batchable/Κ
L0  einsum               I_ap2_μ̃_Κ("i,j,μ̃,Κ;a") = TA::einsum(g_μ̃_μ̃_Κ("μ̃',μ̃,Κ"),
                                                              C_ap2_μ̃("i,j,μ̃';a"), "i,j,μ̃,Κ;a");
```

Each downward step adds exactly one kind of information: P4 adds the DF/PNO factorization, P3 adds
the contraction order, P2/L1 adds the schedule (this node is `t-indep` ⇒ build-once, is
`CELL-BOUND` ⇒ a tiling target, is `batchable/Κ` ⇒ a memory lever), P1 adds the concrete TA call.
Read upward, each level *abstracts away* a resource decision — which is exactly what makes each
level independently analyzable.

---

## 4. The key finding: the MPQC-vs-repro difference is a single pass

Everything `MPQC_EVALUATION.md` established falls out cleanly in this framing: **MPQC and the repro
share L4, L3, and L2 exactly** (same derivation, same DF/CSV, same `optimize` options, same binary
forest, same `TA::einsum` primitive). They differ **only in pass P2 (schedule)**:

| P2 decision | MPQC | repro |
|---|---|---|
| cross-term sharing | runtime `CacheManager` canonical-key collisions | generation-time `eliminate_common_subexpressions` |
| cache lifetime | persistent/volatile entries survive `reset()` across iterations | warm/cold split (`ta_warm_t2`) |
| tiling + pmap | inherits the CSV energies array's trange/shape/**pmap** (`cck.ipp:1567-1600`) | own `build_tot_array` tiling + default/cyclic pmap |
| aux-Κ batching | `make_batched_custom_evaluator` (runtime) | `aux_k_batching.h` (static, on the flat artifact) |

Making L1 a first-class IR — the schedule as data, not as a runtime overlay or a hand edit — turns
this table into a **pluggable P2 pass**: "MPQC-style schedule" vs "repro-style schedule" become two
lowerings of the same L2 forest to the same L1 shape. Concrete payoffs beyond tidiness:
- the CTIR **CELL-BOUND / cost model becomes an *input* that drives** the tiling and batching
  decisions in P2, instead of a post-hoc diagnostic (today tiling is chosen blind, TA-side);
- **L0 becomes retargetable** independently of the schedule (the ~8 existing backends already prove
  the emission boundary is clean);
- each pass is **independently verifiable** (L4/L3 by the residual checksum; L2 by contraction
  count; L1 by the CTIR annotations already validated against MPQC's `cck.ipp`).

---

## 5. Feasibility and a first increment

**Feasible.** The two hardest layers already exist as real, load-bearing types (L4/L3 `ExprPtr`, L2
`EvalNode` forest); the emission boundary is a clean, 8-backend plug-in; and CTIR already proves an
L1-shaped value-DAG is extractable from the forest with the right annotations. The missing pieces
are specific and bounded:
- **the executable L1 Scheduled IR** — CTIR carries the schedule as annotations; promoting them to
  decisions that lower to L0 is the core new artifact;
- **true IR→IR lowering** — nothing today consumes a backend's output; P1 would read L1, not L2;
- **the honest hard part: L1 must *own* tiling/pmap.** Today tiling/distribution is entirely TA-side
  and orthogonal to the SeQuant IR (`build_tot_array`; MPQC inherits it from the solver). For L1 to
  be a real execution plan it has to represent the tiling/pmap decision, which no layer does yet —
  this is the biggest new surface, and the one most worth prototyping to de-risk the idea.

**Recommended first increment (a separate build, scoped here):**
1. **Per-layer textual dumps** at the L4/L3/L2 boundaries — small printers reusing SeQuant
   serialization (L4/L3) and a forest walker (L2) — so the "progressive lowering" is *visible*, not
   just asserted. (L1 and L0 dumps already exist: CTIR and the generated einsum.)
2. **Promote CTIR to an executable L1**: carry tiling/cache/batch as decisions + a minimal
   **L1→L0 lowering** that drives `TiledArrayGenerator` + `build_tot_array` + `aux_k_batching`.
3. **Make P2 a swappable pass** (repro-static vs MPQC-faithful schedule) over the same L2 forest —
   directly operationalizing §4.

This would turn today's "one derivation → parallel backends" into a genuine layered compiler whose
middle level (the schedule) is exactly where the physics-performance decisions — and the entire
MPQC-vs-repro story — actually live.

---

*Grounding: pipeline stages and types verified against `mpqc4/.../cc/{cck.ipp,sequant.cpp}` and
`sequant-fork/SeQuant/core/{eval,optimize,export,domain/mbpt/rules}`; the L1 prototype is the
committed CTIR (`CONTRACTION_IR.md`, `docs/scaling-campaign-data/whole_t{1,2}_residual.ctir`); the
"divergence = one pass" claim tracks `MPQC_EVALUATION.md` §4-6/§8/§10.*
