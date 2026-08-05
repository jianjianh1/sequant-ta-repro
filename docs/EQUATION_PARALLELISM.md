# Equation- and contraction-level parallelism in MPQC's CCSD residual

**Characterization study, 2026-08-05.** Zero machine hours: every number below comes from
artifacts already on disk. No changes to MPQC, TiledArray, or SeQuant.

Scope: how much of the CCSD residual's work *could* run concurrently, how much concurrency
MPQC realizes today, and what specifically prevents more. Equation-level is the focus;
contraction-level appears as the upper bound that shows what equation-level scheduling
cannot reach.

**Data provenance.** Campaign
`/proj/perf-model-gpu-PG0/native-residual-c2-c6-node16-node31-20260804`, build
`mpqc-93593706ed64-sif-b8a53bab5930`, SIF sha256 `b8a53bab5930…`, MADNESS backend PaRSEC,
8 threads/rank, 1 rank/node, node16–node31 (Xeon D-1548, 8 physical cores / 16 SMT,
12 MiB L3, 1 NUMA node). Equation set `5ee7844d…` — 727 binary products, 26 R1 + 55 R2
terms. `cache_imeds=false`, `state=post_solve_converged`, `aux_target_size=0`.
Molecules C2H6–C5H12, cc-pVTZ / cc-pVTZ-RI.

Reproduce with `tools/overlap_analysis.py` and `tools/dag_parallelism.py`.

---

## 1. Headline: MPQC realizes zero contraction-level overlap

Compare the *asynchronous* whole-residual wall (TiledArray's normal scheduling, fenced only
at the ends — `MPQC_NATIVE_RESIDUAL_CSV`) against the strictly serialized sum of that same
residual's own per-product completion walls, from the level-4 graph trace of the same
process, same amplitudes, same `cache=none`.

Define `G = T_async − Σ_products`. If MPQC overlapped products, the async residual would
finish in **less** than the serialized sum and `G` would be negative.

| molecule | R | products | Σ_products (s) | T_async (s) | ratio | G (s) | G% | CV% |
|---|---|---|---|---|---|---|---|---|
| C2H6 | R1 | 180 | 0.575 | 0.618 | 1.075 | +0.043 | 7.5 | 5.61 |
| C2H6 | R2 | 547 | 10.059 | 11.380 | **1.131** | +1.322 | 13.1 | 0.49 |
| C3H8 | R1 | 180 | 1.559 | 1.685 | 1.081 | +0.126 | 8.1 | 1.42 |
| C3H8 | R2 | 547 | 29.884 | 32.973 | **1.103** | +3.090 | 10.3 | 0.63 |
| C4H10 | R1 | 180 | 2.987 | 3.286 | 1.100 | +0.299 | 10.0 | 2.55 |
| C4H10 | R2 | 547 | 83.950 | 92.742 | **1.105** | +8.792 | 10.5 | 0.17 |
| C5H12 | R1 | 180 | 5.358 | 5.724 | 1.068 | +0.366 | 6.8 | 2.20 |
| C5H12 | R2 | 547 | 335.054 | 360.884 | **1.077** | +25.831 | 7.7 | 2.61 |

`G > 0` in **all 8 cells** (ratio min 1.068, median 1.091, max 1.131). The asynchronous
residual costs **6.8–13.1% more** than the strictly serialized sum of its own products.

**The direction is unambiguous** because every measurement artifact pushes the same way:
`Σ_products` *omits* accumulation, symmetrization, and all inter-product gaps, which
deflates the serial side. The comparison is handicapped in favour of finding overlap, and
overlap still is not found.

**The comparison is cross-pass, and that is where its force comes from.** SeQuant's
`evaluate()` is a single-threaded recursion, so within one pass the per-product timed
intervals are disjoint *by construction* and `Σ_products ≤ pass wall` regardless of what TA
does internally — a within-pass comparison could not detect overlap. But `Σ_products` comes
from the traced level-4 export pass (`cck.ipp:2094`, run *after* the timed trials) while
`T_async` comes from the untraced benchmark trials. Tracing can only inflate the per-product
walls, so `T_async > Σ_traced ≥ Σ_untraced`, and the no-overlap direction survives.

**Verdict: MPQC evaluates its residual contractions strictly serially, and pays a further
6.8–13.1% in overhead that is not attributable to any product's own work.**

**This rules out equation-level overlap a fortiori, and §3 explains why it must.** All 81
terms are evaluated by one thread in a sequential loop (`cck.ipp:1813`, then
`for (auto&& n : ranges::views::tail(nodes)) { auto temp = evaluate_term(n); … }`); inside
each term every contraction blocks that same thread at `dist_eval.wait()`; and every
`TA::einsum` opens with a collective fence. No two contractions can overlap, hence no two
equations can. That is structural, not a statistical inference from the table above — which
is why the 6.8–13.1% should be read as dead time layered on strict serialization, not as
partial overlap that failed to pay off.

The per-product walls are completion times, not submission times:
`detail::timed_eval_inplace` (sequant-fork `SeQuant/core/eval/eval.hpp:349-360`) wraps the
TiledArray statement, and TA's expression assignment ends in `dist_eval.wait()` (upstream
`TiledArray/expressions/expr.h:424`), which blocks until the statement's local output tiles
are computed. The `rss()` read and log formatting happen after the timer stops.

### Where the 6.8–13.1% goes — and where it does not

| molecule | R | G% | leaf loads % | permutes % | unexplained % |
|---|---|---|---|---|---|
| C2H6 | R2 | 13.1 | 0.068 | 2.25 | 10.8 |
| C3H8 | R2 | 10.3 | 0.022 | 0.56 | 9.8 |
| C4H10 | R2 | 10.5 | 0.008 | 0.23 | 10.2 |
| C5H12 | R2 | 7.7 | 0.002 | 0.06 | 7.6 |

Leaf loads are negligible. The closing per-term permutations are small. **Accumulation is
not the answer either:** from a genuine `cache_imeds=false` run
(`mpqc-benchmark/work/ethane-cachefree-perf.log`), `Eval | SumInplace` is
19.1 ms against an 11.109 s residual (**0.17%**), and 28.9 ms against 16.456 s on the second
iteration (**0.18%**). That figure is measured with `emit_diagnostics=true`, i.e. with a
`gop.fence()` after *every* `+=` (`cck.ipp:1833`), so it is an **upper bound**.

So 7.6–10.8% of the residual is **inter-product gap** — time inside the residual that
belongs to no product, no leaf load, no permutation, and no accumulation. That is the
per-contraction fence and deferred-cleanup cost, measured end to end at the application
level.

Note the accumulation log is cc-pVDZ-F12 / aug-cc-pVDZ-RI while the campaign is cc-pVTZ, so
it bounds rather than measures the campaign's cells.

**Trial-order control:** last/first trial ratio is 0.957–1.118 across the eight cells and
non-monotone, so the "later passes run slower" confound is bounded at a few percent and has
no trend.

---

## 2. Available parallelism: the ceiling, and what binds it

Structure reconstructed from the SHA-pinned execution plans
(`mpqc-benchmark/catalogs/native-no-cache-v1/execution-plans/*.json`) by stack-simulating
each term's post-order op list, weighted with the measured per-product walls.

**Reconstruction is exact for all four molecules:** 81/81 terms, 727/727 products, and every
popped operand pair matches the Product's printed annotation. Because the plan and the trace
are the same post-order op sequence, this also verifies per-term product counts agree
between the two independent artifacts — a stronger consistency check than an aggregate sum.

**Term independence is verified, not assumed.** The leaf census over all 81 terms is
`C:344, t:154, g:142, s:96, f:10, Constant:62` with **no `I(...)` leaves**: with
`cache_imeds=false` no intermediate is reused across terms, so the 81 terms are mutually
independent and the only cross-term edge is the accumulation into `R`, which is a reduction.

### R2 (the residual that matters — 10–335 s vs R1's 0.6–5.7 s)

| molecule | work (s) | span (s) | work/span | largest term | equation-level ceiling | contraction-level ceiling |
|---|---|---|---|---|---|---|
| C2H6 | 10.059 | 2.573 | 3.91 | R2.T039 = 43.8% | **2.29×** | **3.91×** |
| C3H8 | 29.884 | 7.934 | 3.77 | R2.T039 = 47.9% | **2.09×** | **3.77×** |
| C4H10 | 83.950 | 18.518 | 4.53 | R2.T039 = 41.2% | **2.43×** | **4.53×** |
| C5H12 | 335.054 | 65.598 | 5.11 | R2.T039 = 32.0% | **3.13×** | **5.11×** |

Equation-level = LPT bin-packing of the 55 independent term costs. Contraction-level =
greedy list schedule over the product DAG respecting each term's internal tree. Both
saturate by N=4–8; beyond that there is nothing left to schedule.

R1 is far more even (largest term 17.5–20.7%) and reaches 4.84–5.71× equation-level, but R1
is only 1.6–5.2% of the residual pair's cost (falling as the molecule grows), so it does not
move the headline.

### What binds it: one term, whose two halves are already independent

`R2.T039` is the largest term in **every** molecule — 32–48% of R2's work in only
**6 products**. It is the μ̃Κ half-transform. For C4H10:

| cost (s) | % of term | contraction |
|---|---|---|
| 12.960 | 37.5 | `g(m1,m2,K) * C(i1,i2,m1;a1) -> I(i1,i2,m2,K;a1)` |
| 12.941 | 37.5 | `g(m3,m4,K) * C(i1,i2,m3;a4) -> I(i1,i2,m4,K;a4)` |
| 3.092 | 8.9 | `I(…m4,K;a4) * C(i1,i2,m4;a3) -> I(i1,i2,K;a3,a4)` |
| 2.807 | 8.1 | `I(…m2,K;a1) * C(i1,i2,m2;a2) -> I(i1,i2,K;a2,a1)` |
| 2.211 | 6.4 | `I(…;a1,a3) * I(…;a3,a4) -> R(i1,i2;a4,a1)` |
| 0.540 | 1.6 | `I(…;a2,a1) * t(i1,i2;a3,a2) -> I(i1,i2,K;a1,a3)` |

The two 12.96 s contractions — **75% of the term and 30.9% of all of R2** — are **mutually
independent**: neither is in the other's subtree (verified by ancestor test). They form two
parallel chains that merge only at the final product. The term's internal work/span is
34.551/18.518 = **1.87×**.

**This is the study's central finding.** The binding constraint on equation-level
parallelism is a single term whose dominant contractions are *already* concurrent-ready, but
equation-level scheduling cannot exploit them because both live inside one term. That gap —
2.09–3.13× versus 3.77–5.11× — is entirely accounted for by `R2.T039`.

This term is the same μ̃Κ half-transform that the earlier campaign identified from the other
direction: it is the repro's op-487/488 that `SPTC_SCALE_GEMM` (1.53×@1thr) and
`SPTC_CE_E_GEMM` (1.27×@8thr) targeted with intra-contraction coalescing, and the op that
`MPQC_PROFILE_DEEP.md:78` found **replicated across ranks rather than divided**. Three
independent lines of investigation converge on the same contraction.

### Caveat that must accompany every ceiling above

These are ceilings at **fixed per-worker thread budget**. The weights are 8-thread walls, so
treating N concurrent workers as each having 8 threads over-counts the machine by N×. On
this 8-core hardware, N workers would realistically get 8/N threads each. Converting to a
fixed *total* budget requires `f(t) = wall(t threads)/wall(8 threads)`, which this study
does not measure. Communication, scheduling overhead, and memory/L3 contention between
concurrent workers are all modelled as zero.

**Read these as upper bounds on available parallelism, not as predicted speedups.**

---

## 3. Why the serialization exists: blocker inventory

Every blocker is **upstream**, not a local fork artifact. Provenance was audited rather than
assumed (§4).

| # | blocker | location | provenance |
|---|---|---|---|
| 1 | `TA::einsum` opens with an unconditional **collective** `world.gop.fence()` — once per product | `einsum/tiledarray.h:420` (upstream), `:525` (fork) | upstream |
| 2 | `dist_eval.wait()` blocks the **issuing thread** until the statement's local output tiles are done, so two consecutive statements from one thread never overlap their DAGs | `dist_eval/dist_eval.h:205-222`, `expressions/expr.h:424` | upstream |
| 3 | `+=` makes the LHS an **input** of the RHS expression, so the 55-term accumulation is a serial chain | `expressions/tsr_expr.h:150` — `operator=(AddExpr<TsrExpr_,D>(*this, other.derived()))`; used at `cck.ipp:1570-1572` | upstream |
| 4 | Global sparse threshold set via `gop.serial_invoke` on the default world, called per K-batch from inside the residual | `math/external/tiledarray/threshold.h:19-27`; `cck.ipp:1693-1697` | upstream |
| 5 | `TiledArray::detail::default_world` is a process-global `static World*`, **not** `thread_local` | `TiledArray/external/madness.h:43-70`, with the comment *"this assumes that only 1 thread (usually, main) parses TiledArray DSL"* | upstream |
| 6 | `SafeMPI::unique_tag()` has **no mutex, by design** — all processes must call it in the same sequence, so concurrent fences on one World mismatch tags | `madness/world/safempi.h:521-531` | upstream |
| 7 | `LCAOFactory` memoizing registry, unlocked | `cck.ipp:1523,1531` | upstream |
| 8 | `csv_eqn.df_tensor` / `csv_tensor` mutable `std::map`s written during evaluation | `cck.ipp:1528-1535,1562-1565` | upstream |

Blockers 1 and 2 are the structural pair that explains §1 completely: (2) means a single
issuing thread cannot overlap two statements, and (1) means even multiple issuing threads
cannot help on one World.

**One hazard is absent under this study's contract.** SeQuant's `CacheManager::entry::access()`
(`cache_manager.hpp:88-96`, upstream, 0 fork commits) mutates `life_c` and moves `data_p` out
without synchronization — but with `cache_imeds=false`, `cck.ipp:1751-1752` takes the 3-arg
`sequant::evaluate`, giving each term a fresh stack-local `CacheManager::empty()`. No
sharing, so no hazard.

**A tension worth recording:** blocker 4 means K-batching — the only mechanism that bounds
peak memory (unbatched peak RSS reaches 50.2 GB for C5H12 against a 62 GB node) — is
precisely the mechanism that would forbid concurrency. Any future concurrency work must find
its memory budget elsewhere.

### What is *not* blocked: subworlds

`madness::World::taskq` is a **per-World member** (`madness/world/world.h:206`) while
`ThreadPool` is a **process-wide singleton** (`madness/world/thread.h:1191,1298`). Separate
Worlds therefore have private task queues and private fences over one shared thread pool —
which routes around blockers 1, 5, and 6.

This is upstream-sanctioned, not speculative. Official MPQC ships
`doc/examples/multitask/multitask.cpp`, which is exactly this pattern at whole-wavefunction
granularity:

```cpp
SafeMPI::Intracomm comm = world.mpi.comm().Split(i_am_odd ? 1 : 0, 0);
World my_subworld(comm);
kv->assign("world", &my_subworld);       // default execution context for this input
mpqc::MPQCTask task(my_subworld, kv);  task.run();
my_subworld.gop.fence();                 // "TODO rid of this once madness::World::~World fences"
world.gop.sum(energy, ...);              // combine across subworlds
```

MPQC also builds a per-rank subworld at startup (`mpqc_init.cpp:265-272`, upstream:
`Get_group().Incl(1,&rank)` → `Create(group)` → `new madness::World(comm)` → fence, with
matching fence-before-delete teardown in `~MPQCInit`). The mechanism, its lifetime hazards,
and its MPQC idiom are all already solved upstream.

---

## 4. Fork/upstream attribution audit

The claim "MPQC evaluates its residual serially" must be about MPQC, not about one working
tree. Audited:

| component | finding | evidence |
|---|---|---|
| **MPQC CCk** | The local fork's 71 commits on `batched-tn-eval` touch **only** `cck.h` (69 lines) and `cck.ipp` (764 lines), and are purely additive instrumentation. `cck.h`/`cck.ipp` have 314/381 commits on `origin/master` — CCk is upstream code. | `git diff --stat origin/batched-tn-eval..HEAD -- src/` |
| **the serial accumulation chain** | **upstream** — present at `origin/batched-tn-eval:cck.ipp:1570-1572` | `git show origin/batched-tn-eval:…/cck.ipp` |
| **`threshold.h`, `mpqc_init.cpp`** | byte-identical to upstream | `git diff --quiet origin/batched-tn-eval -- <file>` |
| **TA einsum entry fence** | **upstream**, present at upstream tip `7f76cda0` (v1.1.0-163) | see §5 |
| **`einsum_legacy_subworld`, `einsum_instrument_enabled`, `einsum_differential`, `EinsumBucket`** | **fork-only** — 0 hits anywhere in upstream TA; upstream has no `einsum_instrument.h` at all | `grep -rl … ValeevGroup/tiledarray` |
| **SeQuant `cache_manager.hpp`** | **upstream** (0 fork commits) | `git rev-list --count origin/master..HEAD -- <file>` |
| **SeQuant `eval.hpp`, `result.hpp`** | serial recursion is upstream; the fork adds one commit (`533dca6e`, per-result checksum for trace sanity checks). `448a3960` and `960267df` are on `origin/master`. | `git merge-base --is-ancestor` |
| **pinned TA** | MPQC pins `jianjianh1/tiledarray` at `cd53bd3e04b…` (`external/versions.cmake:6-7`), not ValeevGroup — so fork-only TA gates *are* compiled into the measured SIF | `versions.cmake` |

**Net:** every mechanism that causes the serialization is upstream. The fork contributes the
instrumentation that measures it, plus TA gates that do not affect the default execution
path.

---

## 5. Correction to `MPQC_MULTIRANK.md:126-134`

That section concluded that removing `TA::einsum`'s entry fence "requires reworking
TiledArray's distributed-einsum comm/progress model — a substantial TA-fork change," on the
grounds that the fence guards the distributed subworld-split SUMMA.

**The conclusion stands. The stated reason is stale for the pinned fork, and the real reason
is stronger.**

The fork's `einsum_legacy_subworld()` gate (`einsum/tiledarray.h:48-54`) defaults to
`false`, so the `MPI_Comm_split` path the fence was said to protect is **off** on the fork's
live path: 100% of the 727 products pay the fence and none take that path. Upstream has no
such gate and reaches `Intracomm::Split` at `:798` unconditionally.

But upstream's own history settles it, and it is more decisive than the subworld argument:

- `c052eed0` (2025-03-01, Eduard Valeyev), via PR #500, introduced the fence —
  *"einsum hotfix: pre-fence before entering blocking region, **should not be needed if have
  at least 2 threads to compute with (per rank)** but some compute patterns cause deadlock."*
- `0ce77a9d` **removed** it, on the theory that the MADNESS sticky-task fix
  (m-a-d-n-e-s-s/madness#591) plus wrapping `MPI_Comm_split` in `madness_blocking_invoke`
  sufficed — *"to be tested at scale."*
- `3a21b637` **reverted that removal.** The fence is still present at upstream tip.

**Upstream attempted this exact optimization and had to put the fence back.** It must be
treated as a known upstream serialization with a documented failed removal attempt, not as
available headroom. Note also that the original commit message ties the fence directly to
thread supply, which connects it to the measured thread starvation in
`MPQC_PROFILE_DEEP.md`.

---

## 6. Superseded: ceiling estimates from cache-enabled logs

Earlier ceiling estimates (R2 largest term ≈ 25.3%, pooled LPT 2.00/4.00/5.17×) were derived
by segmenting `mpqc-benchmark/work/ethane-discover-L4.log`. That log is **`cache_imeds:
true`** — its per-pass product counts are 121/238/135/211, not 180/547, because caching and
CSE change the graph. Per `mpqc-benchmark/README.md`, cache-enabled traces are classified
`LEGACY_CACHE_ENABLED_DIAGNOSTIC` and are not valid no-cache measurements.

All ceilings in §2 are recomputed from a single consistent cache-free source. The
cache-enabled figures are superseded and should not be quoted.

---

## 7. Limitations

- **np=1 only.** The level-4 graph trace is emitted by rank 0 and is **not** `gop.max`
  reduced, so at np>1 `Σ_products` understates the max-rank critical path. `overlap_analysis.py`
  prints np>1 rows only under `--all-np` and labels them DIRECTIONAL. No multi-rank claim is
  made here.
- **No variance on the serial side.** One traced pass per cell (`export_after_timing` runs
  once by design). The async side has 3 trials, CV 0.17–5.61%.
- **Ceilings are at fixed per-worker thread budget** (§2 caveat). This is the largest source
  of error in the ceiling numbers.
- **Achievability is not established.** This study bounds what is available and identifies
  what blocks it. Whether concurrency would pay depends on thread scaling and contention,
  neither of which this data measures.
- A same-cell lint in `overlap_analysis.py` refuses any ratio whose numerator and
  denominator differ in molecule, residual, or np — an earlier campaign compared np16 against
  np1 and had to be corrected.

---

## 8. What a follow-on study would do

Ordered by what the findings above actually justify.

1. **Measure `f(t) = wall(t threads)/wall(8 threads)` first.** This is decisive and cheap.
   Every ceiling in §2 is stated at fixed per-worker thread budget; `f` converts it to a
   fixed total budget. If `f(1) ≈ 1.06` — which the prior MPQC thread sweep suggests, cold T2
   1→8 threads being only 8.26→7.80 s — then N single-threaded workers beat one 8-threaded
   worker and the 2–3× equation-level ceiling is reachable. If `f(1) ≈ 3`, equation
   concurrency is a **slowdown**. Do not assert any payoff before this number exists.
2. **Target `R2.T039`, not the 55-term schedule.** §2 shows the equation-level ceiling is
   bound by one term whose two dominant contractions are already independent and are 31% of
   R2. Splitting that one term's two halves is worth more than scheduling the other 54 terms,
   and it is where the equation-level and contraction-level ceilings diverge.
3. **If concurrency is implemented, use rank-partitioned subworlds** (§3), following
   `doc/examples/multitask/multitask.cpp` and `mpqc_init.cpp:265-272`. Subworlds route around
   the three blockers that make thread-level concurrency unsafe (collective entry fence,
   process-global default world, unsynchronized `unique_tag`). Note that per-rank memory, not
   arithmetic, is the binding constraint for any scheme that puts concurrent terms on the same
   rank: unbatched peak RSS is 19.2 GB (C4H10) and 50.2 GB (C5H12) against a 62 GB node, and
   blocker 4 means K-batching cannot be combined with concurrency.
4. **Do not pursue the einsum entry fence** as a bounded lever without engaging upstream
   (§5). Upstream tried and reverted.
