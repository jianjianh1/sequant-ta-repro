# How MPQC evaluates the closed-shell CSV-CCSD residual — and how the repro compares

*Companion to `MPQC_COMPARISON.md`. That document measures the gap; this one traces, end
to end, **what MPQC actually does** to evaluate one T1/T2 residual, cites the exact code, and
for each stage states what the reproduction already matches, what it doesn't, and what
matching would take. All line numbers are against the `mpqc4` and `sequant-fork` trees checked
out on this node (`mpqc4/src/mpqc/chemistry/qc/lcao/cc/cck.{ipp,h}`; SeQuant eval backend under
`sequant-fork/SeQuant/core/`). Both build on the same TiledArray fork commit `cd53bd3`
(`jianjianh1/tiledarray:csv-cck-summa-root-fix`), so any difference is above the TA layer.
`LAYERED_IR.md` recasts this stage-by-stage trace as a formal layered-IR lowering and shows the
MPQC-vs-repro difference is a single scheduling pass.*

> **CORRECTION (2026-08-03) — the §8/§9 cold-gap mechanism is superseded; read it as historical.**
> Later committed measurement (`docs/scaling-campaign-data/gap_profile.txt`, `gap_ceiling.csv`;
> `MPQC_PROFILE_DEEP.md`) refutes the "one contraction is **~87 % of cold T2** and runs **~100× off
> peak** = per-outer-cell ToT tile-task overhead of the **flat×ToT** half-transform (`:487`)" claim
> repeated below (§8 heading, §9, Levers):
> - The wall-dominant op at 8 threads is **`generated_t2_residual.cpp:488`** (the **ToT×ToT**
>   contraction over outer μ̃), and `gap_profile.txt` states "**line 487 (flat×ToT) NOT in top ops**".
>   The top op is ≈**39 %** of the einsum-region (30.5 s of 77.75 s, C4H10 8thr), not 87 %.
> - The achievable ceiling over a hand per-pair GEMM is **≈2.5×** (`gap_ceiling.csv`), and the
>   hand-GEMM itself runs at only ~9-10 % of peak (skinny-GEMM-capped) — there is no "~100×-off-peak"
>   FLOP deficit to recover.
> - The mechanism is **thread-starvation / compute-dispatch-bound** (dgemm ~6 % self-time,
>   `ConditionVariable::wait` ~45 % at 8 threads; IPC 2.55, DRAM ≤23 % of peak — *not* memory-bound),
>   not per-outer-cell tile-task overhead. The `:487` flat×ToT scale is the **1-thread** lever, closed
>   by the landed `SPTC_SCALE_GEMM` (`MPQC_SINGLE_THREAD.md`).
> - **The Stage-8 "pmap/tiling" lever and the "flat×ToT per-cell fix / per-pair proto domain" remedies
>   are refuted** (§8's own sweep shows no speedup; `cyclic_pmap_timing.csv`). The actual lever is the
>   runtime evaluator's work-coalescing + solver-inherited layout — a *naive* `sequant::evaluate` port
>   was built and measured **1.2× slower** (`MPQC_RUNTIME_EVAL.md`); the unified verdict is in
>   `MPQC_PROFILE_DEEP.md`.

## The one-line story

MPQC and the repro run the **same algebra** (SeQuant-derived, density-fitted, CSV-transformed
closed-shell CCSD) and issue the **same primitive** (`TA::einsum` / `einsum<DeNest::True>` on
tensor-of-tensor arrays) on the **same TiledArray**. They differ in exactly two places that
matter for performance:

1. **MPQC walks a per-term binary-tree forest through a runtime interpreter with a
   `CacheManager`**; the repro replays a **flattened, statically-generated straight-line
   sequence** of the same contractions. For a *single* residual evaluation these are
   equivalent — the repro's generation-time CSE ≈ MPQC's runtime cache, and the repro's
   warm/cold split (`ta_warm_t2`) ≈ MPQC's persistent-vs-volatile cache entries across solver
   iterations. **This is effectively matched** and is *not* the source of the measured gap.
2. **MPQC's residual arrays inherit the CSV-machinery's `TiledRange` / `SparseShape` / `pmap`**
   (`cck.ipp:1563-1565`), so the big DF half-transform intermediate is tiled and *distributed*
   the way the upstream PNO/CSV solver already laid out the occupied space. The repro builds
   its **own** tiling and takes **TiledArray's default pmap** in `build_tot_array`
   (`src/ta_builder.h:430-431`, the pmap slot defaulting to empty). **This looked like the
   actionable cold-gap lever, but §8 empirically refutes the input-pmap/tiling fix** — the real
   cost is inside the ToT-`einsum` (see §8 and lever (a)).

A refinement that corrects the earlier framing: in the alkane runs measured for
`MPQC_COMPARISON.md` §11, MPQC's aux-Κ batching was **off** (`batch:aux_target_size` default
`0`, `cck.h:105`, `cck.ipp:247-248`). So **MPQC also materializes the whole giant μ̃Κ DF
half-transform intermediate** — its runtime cache avoids *recomputation*, not *size*. The cold
np=16 gap is therefore **not** "MPQC never forms the intermediate"; both sides form it and both
call the identical `TA::einsum` SUMMA. The gap is in *how that one contraction is evaluated* —
first hypothesized as **tiling + pmap**, but §8 shows the input pmap/tiling is empirically refuted
(cyclic pmap = no speedup) and the real cost is the ToT-`einsum` per-outer-cell overhead (§8/§9).
Aux-Κ batching (§10) is a *separate, opt-in memory* technique — the lever for the
hexane wall, not the np=16 speed.

---

## Stage 1 — Derivation of the equations (identical: both are SeQuant)

`CCk::generate_csv_closedshell` builds the residual equation set
(`cck.ipp:1513-1523`); `evaluate_csv_closedshell` (`cck.ipp:1584`) invokes it once via the
`if (csv_eqn.Rs.empty())` guard (`cck.ipp:1591`):

```
csv_eqn.Rs = make_cceqvec_csv_closedshell(k_, zero_t1_);   // 1513
  ... per equation e:
  e = tail_factor(e);                    // 1519  strip the leading Symmetrizer (S) operator
  if (df_) e = impl::density_fit(e);     // 1520  insert the DF/RI aux index Κ  (impl at cck.ipp:57)
  e = impl::csv_transform(e, csv_->basis()); // 1521  project to the cluster-specific-virtual basis (cck.ipp:62)
  sequant::flatten(e);                   // 1523  flatten nested sums to a flat summand list
```

**Repro: identical.** These arrays are produced by the SeQuant CSV-CCSD generator
(`sequant-fork`, branch `csv-ccsd-tiledarray-generator`,
`tests/manual/test_csv_ccsd_derivation.cpp`) using the *same* `make_cceqvec_csv_closedshell`
→ `tail_factor` → `density_fit` → `csv_transform` → `flatten` pipeline, then emitted as C++
(`src/generated_t1_residual.cpp`, `src/generated_t2_residual.cpp`). Same equations, same DF,
same CSV projection.

## Stage 2 — Per-summand optimization (matched)

Each flattened summand is optimized **independently** — there is **no cross-term CSE at
optimize time**. `optimize()` dispatches to `single_term_opt` under a flop cost model
(`optimize.cpp:37-44`):

```
if (opts.opt_for == OptFor::Flops)
  return opt::single_term_opt<OptFor::Flops>(
      prod, opts.idx_to_extent, subnet_cse, opts.is_volatile_leaf, opts.n_replay);
```

Three knobs shape the factorization:
- **`idx_to_extent`** — per-index extents (incl. the average CSV/PNO extent) size the ToT
  contractions so the cost model picks a realistic contraction order.
- **`is_volatile_leaf`** (the amplitude label `t`) + **`n_replay`** — bias the flop model to
  hoist amplitude-*independent* work (the DF/CSV machinery) out of the amplitude-dependent
  path, so it can be cached/precomputed. This is what makes a warm/cold split meaningful.
- `subnet_cse` — intra-term (within a single summand's binary tree) common-subexpression
  reuse only.

**Repro: matched.** The generator invokes `optimize()` with the same `OptFor::Flops`,
`is_volatile_leaf = (label == t)`, `n_replay`, and CSV-extent options; the emitted C++ is the
result of exactly this factorization.

## Stage 3 — Summand reordering (matched)

`ReorderSum::Reorder` (`optimize.cpp:167`, dispatched via `sum.cpp:113-131` `reorder(Sum
const&)`) permutes the summand list so terms sharing an intermediate are adjacent, shortening
that intermediate's live-range for the runtime cache (relevant only with `cache_imeds`).

**Repro: matched.** The generator uses `ReorderSum::Reorder`; the emitted statement order is
the reordered one.

## Stage 4 — Binarization into a per-term forest (repro flattens this)

MPQC binarizes **each summand** into a `FullBinaryNode<EvalExpr>` tree via `ResultExpr`
(`cck.ipp:618-620`, `:823` "Build per-(rank, spin-block) head templates and binarize each
summand"). The residual equation carries a **forest** `cc_eqn.Ns` — one tree per term
(consumed at `cck.ipp:1281`, `sequant::evaluate(cc_eqn.Ns.front(), leaf_eval)`).

**Repro: flattened.** The generator lowers that same forest to a **static straight-line
sequence** of C++ TA statements (one per binary node, in the reordered order). The tree
structure is compiled away; the arithmetic and the intermediate reuse are preserved. This is
the first of the two real structural differences — but see §6: for one evaluation it is
behavior-equivalent.

## Stage 5 — Runtime interpretation → one `TA::einsum` per node (matched primitive)

MPQC evaluates a term by walking its tree in a runtime interpreter,
`sequant::evaluate(node, leaf_eval, cache)` (`eval.hpp:503`, and the `Nodes` overload at
`:755`/`:814` that accumulates a whole summand list). The residual driver builds
`leaf_eval = make_csv_leaf_eval()` once (`cck.ipp:1594`) and drives each term through an
`evaluator` lambda calling `sequant::evaluate(node, annot, leaf_eval, csv_eqn.Cs[R])`
(`cck.ipp:1679-1690`), where `csv_eqn.Cs[R]` is the term's cache (and the install point for the
batched custom evaluator, §10):
- **leaf** → `make_csv_leaf_eval` binds the node to its TA array (the loaded DF/CSV/amplitude
  tensor);
- **internal `Product` node** → exactly **one** `TA::einsum` call. In the TiledArray backend
  (`backends/tiledarray/result.hpp`): flat×flat and ToT×flat go through
  `TA::einsum(...)` (`result.hpp:381`, `:613`, `:628`); ToT×ToT with de-nesting goes through
  `TA::einsum<TA::DeNest::True>(...)` (`result.hpp:621`);
- **`Sum` node** → the operands are accumulated.

The top-level Σ over summands is accumulated in the driver against a template-pinned output
annotation (`cck.ipp:1706-1708` `make_R_template_csv`).

**Repro: matched primitive.** Every generated statement is one `TA::einsum` /
`einsum<DeNest::True>` on the same operand types with the same annotations — the identical TA
call the interpreter would emit for that node. (Confirmed in `MPQC_COMPARISON.md` §5: same TA,
same operations.)

## Stage 6 — The `CacheManager` (matched, in two pieces)

`CacheManager` (`cache_manager.hpp`) is where MPQC's *runtime* evaluator earns its keep:
- Intermediates are keyed by their **canonical `EvalNode`**; `min_repeats = 2` means only
  reused intermediates are cached. Cross-term sharing that the repro would have to find at
  generation time is realized here at **runtime** by canonical-key collisions across the
  independently-optimized summands.
- Entries are classified **Persistent (P)** vs **Non-persistent (NP)** (`cache_manager.hpp:75-112`).
  NP entries are drained after last use and cleared by `reset()` (called per term,
  `eval.hpp:234`); **P entries survive `reset()`** and are reused **across solver iterations**.
  P = the amplitude-*independent* DF/CSV machinery (the non-volatile subtree feeding the
  volatile `t` leaves) — built once, reused every iteration.

**Repro: matched in two pieces.**
- The within-one-evaluation cross-term sharing ≈ the repro's **generation-time cross-term
  CSE** (the generator emits shared intermediates once). For a single residual, static CSE and
  the runtime cache produce the same reuse.
- The **P/NP cross-iteration** reuse ≈ the repro's **warm/cold split**: `ta_warm_t2`
  (`tools/gen_split/`) precomputes the t-independent DF/CSV block once and times only the
  t-dependent update — exactly MPQC's P-entries-survive-`reset()` behavior. Warm-vs-warm they are
  near parity for the small molecules (C₃H₈ cc-pVTZ warm np1 ≈ 1.3×, `MPQC_COMPARISON.md` §11;
  the ratio grows to ~2–3× by C₅H₁₂ — §11's headline "repro ~2–3× slower warm"), which is the
  evidence this stage's *mechanism* is matched even though the wall-clock diverges with size.

## Stage 7 — Result template + R2 symmetrization (matched as-is)

`make_R_template_csv` (`cck.h:507`) pins the residual's ToT layout (outer = occupied-index
tuple, inner = per-pair PNO/proto indices) and the literal slot order used when accumulating
summands (`cck.ipp:1549`, `:1706`). After accumulation, R2 is **symmetrized**
(`cck.ipp:1754`):

```
result("i,j;a,b") = 0.5 * (result("i,j;a,b") + result("j,i;b,a"));
```

**Repro: matched as-is (raw R2 is the reference).** The repro's raw R2 is what reproduces the
validated reference; its R2 is in fact ~antisymmetric under that (i↔j, a↔b) swap, so
symmetrizing *diverges* from the reference rather than matching it. This was tested directly
via the env-gated `SPTC_SYMMETRIZE_R2` knob
(`src/ta_sequant_native_residual_main.cpp:62-68`) and is documented in `MPQC_COMPARISON.md`
§11 gap-closing attempts. So the repro correctly does **not** symmetrize; nothing to change.

## Stage 8 — ToT tiling + pmap (hypothesized cold-gap lever — **empirically refuted**, see §8 + the 2026-08-03 correction box)

This is the load-bearing difference. When MPQC builds a CSV residual/amplitude ToT array, it
does **not** choose a fresh tiling — it **inherits the CSV energies array's exact
`TiledRange`, `SparseShape`, and `pmap`** (`cck.ipp:1563-1565`, in `zero_init_csv_amplitudes`):

```
const auto& energies_t = csv_->energies(t);
T_csv_[t - 1] = ArrayToT(energies_t.world(), energies_t.trange(),
                         energies_t.shape(), energies_t.pmap());   // inherit layout + pmap
```

The occupied-space tiling itself is set upstream in the orbital registry — the residual's occ
`TiledRange1` is `orb_reg.retrieve("i")->trange()` (`cck.ipp:574`), fixed by the LCAO/PNO
machinery, **not** a literal constant in `cck`. The consequence: every DF-carrying residual ToT
shares one occupied-space tiling *and one process map* with the arrays that produced it, so the
SUMMA of the giant μ̃Κ half-transform (§9) is laid out — and load-balanced across ranks — the
way the solver already decided.

**Repro: builds its own tiling and takes TiledArray's default pmap.** `build_tot_array`
(`src/ta_builder.h:211-449`) constructs the outer `TiledRange` from env-driven knobs
(`SPTC_TILES_PER_DIM` at `ta_builder.h:60-62`, `SPTC_COARSE_OCC` / `SPTC_OCC_TILE` at `:293-298`;
the `make_trange` build at `:380-390`) and then
`ArrayToT array(world, outer_trange, sp_shape, maybe_cyclic_pmap(...))` (`ta_builder.h:430-431`) —
the pmap slot **defaults to empty** (TA's default blocked pmap keyed off that trange) unless
`SPTC_CYCLIC_PMAP` is set (`maybe_cyclic_pmap`, `:146-151`). Two arrays that MPQC
would co-locate can land on different rank layouts here. Since the cold np=16 cost is dominated
by the SUMMA of the one giant DF intermediate (§9; ~87% of cold T2 per the campaign per-op trace,
not committed here — **superseded: op :488 ≈40 %, see the top correction box + `gap_profile.txt`**),
its proc-grid and load balance — i.e. this tiling+pmap — is the plausible
cold-gap cause. **Actionable** (see §Verification and §Levers) — and the verification below
narrows *which* part: the occ tile *size* turns out not to matter, so it is the **pmap
co-location**, not the tiling granularity, that is the real lever.

## Stage 9 — Distribution of the DF half-transform (same TA/SUMMA)

The big intermediate `I[i,i,μ̃,Κ;a] = g_μ̃μ̃Κ (dense DF) × C_ap2_μ̃ (ToT)` and its consumer are
`TA::einsum` calls whose distributed contraction is TiledArray's **SUMMA** over the operand
pmaps. The pinned fork commit `cd53bd3` ("drop the `BinaryEvalImpl` trange-equality assertion,
K-batch friendly", exercised by the batched-evaluator install at `cck.ipp:1643`) is what lets *sliced* tranges drive SUMMA —
required for the batched path (§10) and for the sparse/ragged tranges these ToT arrays carry.

**Repro: same TA, same SUMMA.** No difference at this layer — which is precisely why §8 (what
pmap the operands carry *into* SUMMA) is where the cold gap lives.

## Stage 10 — Aux-Κ batching (**THE hexane-memory lever, opt-in — off in the measured runs**)

MPQC can stream the DF aux index Κ in tile-aligned slices so the full-Κ intermediate is
**never materialized**. Enabled only when `batch:aux_target_size > 0` (`cck.ipp:1601`,
`batch_aux`); it installs `sequant::make_batched_custom_evaluator` (`eval.hpp:1129`) as the
term's custom evaluator (`cck.ipp:1643`). Mechanics (`cck.ipp:1601-1645`):
- Κ is sliced by `mode_batches_of_trange1` (`result.hpp:249`) into tile-aligned batches;
  partial contractions are summed, bounding peak memory to one batch's worth.
- The block-sparse **screening threshold is divided by `n_batches`** (`cck.ipp:1620-1640`):
  a block kept over the full Κ has ~1/n of its norm-bound in each 1/n-of-Κ sub-sum, so without
  scaling it would be screened away in every batch and its contribution lost. Set/restored
  MPI-collectively via `push_ta_sparse_threshold`.
- **Persistence-gated**: only *persistent* (amplitude-independent, build-once) Κ-contractions
  are batched (`is_volatile` = label `t`, `cck.ipp:1637-1642`); volatile t-dependent ops live
  in the small compressed CSV space, so batching them is pure recurring overhead.

**Crucially, `batch:aux_target_size` defaulted to `0` in the alkane runs** (`cck.h:105`,
`cck.ipp:247-248`) — so **MPQC also formed the whole giant intermediate**. Batching is not the
reason MPQC is faster at np=16; it is the principled way to fit a problem whose intermediate
does not fit in memory.

**Repro: implemented and validated** (2026-07-30). Ported as `src/aux_k_batching.h`
(`accumulate_df_halftransform_batched`), gated by the env var `SPTC_AUX_TARGET_SIZE` (target Κ
elements per batch, 0 = off — the same knob shape as MPQC's `batch:aux_target_size`), compiled
into `ta_auxbatch_main` via the variant TU `src/generated_t2_residual_auxbatch.cpp`. It streams Κ
over the DF half-transform block (`generated_t2_residual.cpp:487-491`); because Κ is contracted at
the block root, the per-batch partials simply sum (`+=`) into the residual — the same 1/`n_batches`
threshold scaling as MPQC, algebraically identical (same cell count, same flops; results agree to
10–13 sig figs modulo Κ-sum reassociation). One implementation
note: a standalone TA `.block()` slice of a sparse array deadlocks / trips "RMI thread not running"
on the pinned `cd53bd3`, so the Κ-slice of `g` is built by an explicit tile copy
(`slice_g_over_K`), not a block expression.

Validated single-node (node3) A/B, `SPTC_AUX_TARGET_SIZE` 0 vs 96, on C₂H₆–C₅H₁₂: T2 checksums
match the unbatched path to 10–13 significant figures (floating-point reassociation of the Κ sum;
`nnz` identical), and **peak RSS drops sharply** — 3.65→2.07 GB (C₂H₆), 12.0→5.0 (C₃H₈),
28.9→10.8 (C₄H₁₀), 56.9→27.0 (C₅H₁₂), i.e. −43 to −63%. So batching is numerically transparent and
bounds exactly the giant intermediate as intended. It is a **memory** lever, not a speed one (the
per-cell ToT-einsum cost is unchanged; §11 ratios are unaffected). Raw data:
`scaling-campaign-data/auxbatch_correctness.csv`. The hexane outcome — where this lever is meant to
pay off — is in `MPQC_COMPARISON.md` §11.

## Stage 11 — Energy and amplitude dump (context, not a perf lever)

The converged residual feeds `evaluate_csv_energy`; the SeQuant selected-trace amplitude dump
is gated by `PostSolveDumpGuard` (`cck.ipp:1054`) and the
`sequant:trace:selected:amps_post_solve` flag (`cck.ipp:293-294`). This flag is the source of
the correctness-anchor off-by-one noted in `MPQC_COMPARISON.md`/campaign memory: the dumped
amplitudes are `T_final`, one update past the last *logged* per-iteration residual, so the
repro's residual matches no single logged occurrence exactly (though `nnz` matches, and the
gauge-free R(T=0) = occurrence 1 does match). Not a performance stage; included so the pipeline
is complete end to end.

---

## Verification experiment — does occ tiling / pmap actually move cold np=16?

To keep §8 an actionable claim rather than an assertion, the repro's occ tiling was swept on
**C₄H₁₀ (active occ extent = 17) cold at np=16** on nodes 16–31, owning-ToT binary, using
`scripts/occ_tiling_experiment.sh` (in the campaign workdir). Note the committed §11 sweep used
`SPTC_COARSE_OCC=9`, which is ethane-specific and never matches C₄H₁₀'s extent 17 (so its occ
dims silently fell back to adaptive ~2); this experiment sets `COARSE_OCC=17` correctly and
compares whole-T2 wall time across occ tilings.

**Result — occ tile *size* is not the lever; it does not close the cold gap.** Whole-T2 cold
wall time at np=16 on C₄H₁₀ (median of 2 timed trials; MPQC's number for the same point is
~10.2 s):

| config | occ tiling | T2 cold np=16 (s) | note |
|---|---|---|---|
| `baseline_committed` (`COARSE_OCC=9`) | adaptive ≈2 (9 never matches C₄H₁₀'s extent 17) | **~48.7** | the committed §11 C₄H₁₀ point |
| `occ17_t2` | occ tiled at 2 (correct extent) | ~49.4 | ≈ baseline — no change |
| `occ17_t4` | occ tiled at 4 | ~52.4 | slightly *worse*; checksum also drifts (see below) |
| `occ17_t8` | occ tiled at 8 | ~80.0 | markedly *worse* |
| `noocc_tpd8` | pure default: pair-key dims forced to size 1 (`ta_builder.h:386-389`) | pathological — did not finish (>14 min/pass, stopped) | why the campaign coarsens occ at all |
| `occ17_t4_tpd6` | occ 4 + `TPD=6` (warm optimum) | ~53.5 | no speedup, and checksum diverges 100× (see caveat) |

Finer occ tiling (2) leaves cold np=16 unchanged (~48–49 s); coarser (4, 8) makes it *worse*;
the pure size-1-pair-key default is pathologically slow. **None approaches MPQC's ~10 s.** So
the occupied-space tile *size* is not the cold-gap lever.

**Direct confirmation — the default pmap leaves ranks idle.** A second diagnostic
(`SPTC_DUMP_PMAP`, `src/ta_builder.h` `dump_pmap_distribution`) dumps each array's nonzero-tile
count per rank at np=16 (`docs/scaling-campaign-data/pmap_distribution_C4H10_np16.txt`; per-rank
pmap distribution is node-count-independent, so it runs `mpirun -np 16` on one node). The result
is unambiguous: TA's **default blocked pmap starves ranks 0, 1, 2 — they own zero tiles of every
*sparse* DF/CSV array** (`g`, `g1`, `c1`, `c2`, `c2_tot`, `t_i_i_a_a`, …), while only the fully
*dense* arrays (`g0`, `f_m_m`, `s_m_m`) are balanced. The mechanism: a contiguous-block pmap maps
the low tile-ordinals to the low ranks, and the frozen-core-zeroed *leading* tiles are exactly
those low ordinals — so ranks 0–2 get all-empty ranges. The giant DF half-transform intermediate
is built from these sparse operands, so its SUMMA runs with **~3/16 (~19%) of ranks idle**,
independent of occ tile size (which is why the sweep above saw no tile-size effect).

This **sharpens §8's claim**: it is not "match MPQC's occ tiling" in general — MPQC's advantage
is the **pmap co-location** of the DF-carrying operands (they inherit the CSV solver's `pmap`,
`cck.ipp:1565`), i.e. *where the giant intermediate's tiles land across ranks and whether the
two SUMMA operands are co-resident*, not how finely the occupied axis is cut. Changing the tile
size (which is all `build_tot_array` currently exposes) reshuffles TA's default blocked pmap but
does not co-locate the operands the way MPQC's inherited pmap does. The actionable lever is
therefore narrower and more specific than first stated, and the idle-rank diagnostic pins the
exact fix: **replace TA's default blocked pmap for the DF-carrying arrays with one that
distributes the *nonzero* tiles evenly across all ranks** (e.g. a cyclic/round-robin pmap over
occupied tiles, or dropping the frozen-core-zeroed leading tiles from the `TiledRange` so the
blocked pmap no longer front-loads empty ranges) — passed explicitly to `ArrayToT array(world,
outer_trange, sp_shape, pmap)` in place of the current defaulted slot at `ta_builder.h:430-431`. This would
engage the ~3 idle ranks; it is *not* an occ-tile-count retune.

*Caveat surfaced by the sweep — coarse multi-pair occ tiling is numerically unsafe here.* With
`COARSE_PAD=0` (ragged) and a genuinely multi-pair occ tile (`OCC_TILE ≥ 4`), the T2 checksum
drifts: `occ17_t4` gives `sum -7.10588 → -7.10768` (~0.03%, the same ragged-ToT order-sensitivity
documented for proto=100 in `MPQC_COMPARISON.md` §11), but adding the warm-optimal `TPD=6` on top
(`occ17_t4_tpd6`) blows the drift up to `sum -0.06725` — a **100× divergence**, reproduced exactly
across two independent invocations (baseline in the *same* invocation stays bit-identical at
`-7.10588`, ruling out corruption). The default `COARSE_OCC=9` config used for the committed §11
numbers never coarsens C₄H₁₀'s occupied axis (9 ≠ extent 17), so it never triggers this — but the
result shows coarse occ tiling is not merely "no speedup," it is a correctness hazard once occ
tiles actually straddle multiple pairs, exactly the constraint `build_tot_array` documents at
`ta_builder.h:287-310`. Two more reasons not to pursue occ retuning; the lever is pmap. (Full
data committed under `docs/scaling-campaign-data/`: `scripts/occ_tiling_experiment.sh` and
`occ_tiling_experiment.csv`.)

**Lever (a) implemented and validated at the input-distribution level (`SPTC_CYCLIC_PMAP`).**
The fix from the idle-rank diagnostic — replace the default blocked pmap with a cyclic
(`TA::detail::RoundRobinPmap`, tile ord → ord % nproc) one for the DF/CSV arrays — is now an
env-gated option in `build_tot_array`/`build_sparse_array` (`maybe_cyclic_pmap`, `ta_builder.h`;
empty/default unless the knob is set, so the default build is unchanged). Two checks confirm it
does what §8 predicted:
- **Correctness (checksum invariance).** A pmap only changes which rank *owns* a tile, never a
  tile's value, so a correct pmap swap must leave the residual identical. Verified: C₂H₆ np=2 T2
  `sum` default `-0.000677185440882994` vs cyclic `-0.000677185440888003` — equal to ~13 sig figs
  (the ~5e-15 tail is just cross-rank reduction reordering), with `nnz`/`sumsq`/`max_abs`
  bit-identical.
- **Behavior (idle ranks eliminated).** C₄H₁₀ np=16: the sparse array `g1` goes from
  `[0,0,0,4,41,25,…]` (3 idle ranks) under the default pmap to `[25,25,25,…,24]` — **0 idle,
  evenly 24–25 tiles/rank** — under the cyclic pmap; every other sparse DF/CSV array balances the
  same way.

**Timing payoff — measured, and it does *not* help (the informative negative result).** A real
multi-node run (owning-ToT binary on nodes 16–31, C₄H₁₀ cold np=16, committed §11 config, cyclic
off vs on; `docs/scaling-campaign-data/cyclic_pmap_timing.{sh,csv}`) shows **no speedup**:

| pmap | T2 cold np=16 (s), 3 trials | median | T2 `sum` |
|---|---|---|---|
| default (blocked) | 50.8 / 48.9 / 47.1 | **48.9** | −7.10587645551 |
| cyclic (`SPTC_CYCLIC_PMAP`) | 49.9 / 84.4 / 49.8 | **49.9** | −7.10587645551 |

Same to slightly worse (one 84 s outlier), still ~5× MPQC's ~10 s; the checksum is invariant at
np=16 too (agree to ~11 sig figs), re-confirming correctness. **So eliminating the idle *input*
ranks does not translate into a faster contraction** — exactly the "balanced input distribution
≠ balanced SUMMA" caveat, now empirically confirmed. TA's `einsum` re-maps operands into its own
SUMMA layout, so the input array pmap is not what governs the giant intermediate's execution.
The cold-gap bottleneck is *inside* the ToT `einsum` — consistent with the earlier per-op profile
(`MPQC_COMPARISON.md`/campaign notes: one contraction is ~87 % of cold T2 and runs ~100× off
peak = per-outer-cell ToT tile-task overhead, not flops, not distribution — **superseded: op :488
≈40 %, ~2.5× ceiling, thread-starvation; see the top correction box**). **Lever (a) as
"fix the input pmap" is therefore refuted as a timing fix.** `SPTC_CYCLIC_PMAP` is kept as a
correct, gated diagnostic (it does balance the inputs), but the real cold cost is the ToT
`einsum`'s per-outer-cell overhead for the giant DF half-transform intermediate itself.

### Can the generator avoid the giant intermediate? (refactor attempt — no)

The natural next idea is to change the *generated factorization* so the DF half-transform never
forms the `(μ̃,Κ)`-both-outer, tiny-per-pair-cell intermediate `I_ap2_μ̃_Κ[i,i,μ̃,Κ;a]`
(`src/generated_t2_residual.cpp:487`). A read of the SeQuant generator settles it:

- **`(μ̃,Κ)`-inner is structurally impossible.** The ToT outer/inner split is a hard function of
  `Index::has_proto_indices()` — inner ⇔ proto-decorated (per-pair PNO domain) — in both the
  emitter (`SeQuant/core/export/tiledarray_generator.hpp:575-576`) and the cost model
  (`SeQuant/core/utility/indices.hpp:405-406`). The PNO index `a` is per-pair (proto ⇒ inner);
  μ̃ (PAO) and Κ (DF aux) are **global, non-proto** (`domain/mbpt/rules/df.cpp:18-63`;
  `csv.cpp:31` even asserts aux is never proto) ⇒ they are forced **outer**. No optimize option
  or extent value can move a non-proto axis into the per-pair inner cell.
- **No correctness-safe optimizer setting reduces the intermediate — they all make it worse.**
  Measured on the regenerated R2, count of `(μ̃,Κ)`-both-outer tiny-cell intermediates:
  `OptFor::Flops` (shipped) **2**; `OptFor::Memsize` **3**; `Flops`+`SPTC_NO_CSE=1` **4** (so
  cross-term CSE actually *merges/reduces* them). The Flops cost model
  (`SeQuant/core/optimize/single_term.hpp:54-56`) has no per-cell/block-overhead term, but adding
  memory pressure (`Memsize`) steers *toward* more small-inner ToT intermediates, not away. The
  `SPTC_OPT_MEMSIZE` knob added to the generator's derivation test documents this (kept, gated,
  `OptFor::Flops` stays default).
- **The one setting that removes them is the already-rejected one.** A large proto extent
  (`SPTC_PROTO_EXTENT=100`) drops the count to **0**, but it does so by swapping in a more-expensive
  μ̃-family factorization that is *slower* with owning-ToT and numerically divergent at cc-pVTZ
  (`MPQC_COMPARISON.md` §11 / campaign memory). There is no free lunch: the `(μ̃,Κ)`-outer
  tiny-cell half-transform *is* the flops-optimal factorization for this DF+CSV algebra.

**Conclusion.** The cold gap is not closable at the generator level. It is a genuine TiledArray
flat×ToT `einsum` efficiency limit for the DF half-transform, and the only real remedies are (i) a
**TA backend improvement** to the flat×ToT contraction's per-cell overhead, or (ii) a
**derivation-level** change that gives the PAO index μ̃ a per-pair (proto) domain so the
half-transform becomes a per-pair quantity (a method change to the DF/CSV rules, not a generator
knob), or (iii) aux-Κ batching to bound *memory* (lever b — it does not address per-cell speed).

---

## Levers to actually close the gap

Naming the three concrete levers (superseding `MPQC_COMPARISON.md` §11's vague "match MPQC's
evaluation approach"):

- **(a) Cold np=16 (~5× at cc-pVTZ) → pmap co-location of the DF-carrying ToT arrays.** Both
  sides materialize the giant μ̃Κ intermediate and call the same `TA::einsum` SUMMA; MPQC's
  operands inherit the CSV solver's `trange`/`shape`/`pmap` (`cck.ipp:1563-1565`), the repro's
  take TA's default from a self-chosen tiling (`ta_builder.h:430-431`). The verification sweep above
  shows the occ tile *size* is **not** the lever (finer = no change, coarser = worse, none near
  MPQC's ~10 s); what remains is the **pmap** — whether the two SUMMA operands are co-resident
  and how the intermediate's tiles spread across ranks. Tried and **refuted** as a timing fix:
  `SPTC_CYCLIC_PMAP` (`maybe_cyclic_pmap`, `ta_builder.h`) balances the input arrays
  (checksum-invariant, 0 idle ranks) but does **not** speed up cold np=16 (48.9 s → 49.9 s; §
  verification above) — TA re-maps operands into its own SUMMA layout, so the input pmap doesn't
  govern the contraction. The genuine cold cost is the ToT `einsum`'s per-outer-cell overhead for
  the giant DF half-transform (~87 % of cold T2, ~100× off peak — **superseded: thread-starvation on
  op :488, ≈40 %, ~2.5× ceiling; see the top correction box + `MPQC_PROFILE_DEEP.md`**). A **generator refactor was also
  tried and refuted** (§8): `(μ̃,Κ)`-inner is structurally impossible (inner ⇔ proto; μ̃,Κ are
  non-proto/global), and no correctness-safe optimizer setting reduces the tiny-cell intermediate
  (`OptFor::Memsize` and `NO_CSE` produce *more*; only proto=100 removes it, at the cost of a
  slower, divergent μ̃-family factorization). So this is **not** an in-repo lever: it needs a TA
  backend fix to flat×ToT contraction, or a derivation-level change giving μ̃ a per-pair proto
  domain — a method change, not a generator/tiling/pmap knob.
- **(b) Hexane memory wall → aux-Κ batching — DONE (memory).** Implemented as
  `src/aux_k_batching.h` (`SPTC_AUX_TARGET_SIZE`): streams Κ in tile-aligned slices over the
  *persistent* DF terms, sums partials, scales the sparse threshold by 1/`n_batches` (mirroring
  `cck.ipp:1601-1645`), on the sliced-trange SUMMA path already in `cd53bd3`. Validated
  transparent (checksums agree to 10–13 sig figs) with peak RSS −43…−63% (Stage 10). With
  regenerated complete leaves it lets **repro hexane complete** single-rank (T1 nnz 1785 / 53.8 s,
  T2 nnz 591501 / 933 s, peak 27.5 GB; `MPQC_COMPARISON.md` §11). The memory fix, independent of (a).
- **(c) Warm (~2×) → mostly fundamental.** ~7% is recoverable with `SPTC_TILES_PER_DIM=6`
  (the warm tiling optimum); the rest is distributed across ~250 small ragged-ToT ops with no
  single-op lever — the intrinsic cost of the flattened static sequence vs the runtime
  evaluator, addressable only by a representation/backend change.

**Bottom line.** The repro matches MPQC on derivation, factorization, the einsum primitive, and
(via static CSE + the warm/cold split) the cache's effect on a single evaluation. The remaining
equal-rank gap is not "MPQC skips work the repro does" — it is **how the identical DF
contraction is tiled and distributed** (a, in-repo) and, for hexane, **whether the intermediate
is streamed** (b, opt-in). Lever (c)'s residual is the one genuinely-structural piece.

> Much of the analysis above (the CELL-BOUND DF half-transform, the `t-indep`/`t-dep` split, the
> persistent build-once intermediates, aux-Κ batchability) is exactly what a static `TA::einsum`
> call list *cannot* show. `docs/CONTRACTION_IR.md` defines **CTIR**, a human-readable Contraction
> IR whose emitter annotates each contraction with these facts at generation time — see it for the
> `I_ap2_μ̃_Κ` node rendered with its `⚠ CELL-BOUND (1.6e6 tiny ToT tasks)` / `t-indep` flags.
