# Cold-time gap: root cause, ceiling, and where the leverage really is

*Offline research (2026-07-30, node3 single-rank unless noted). Replaces the previously
uncommitted "~87% is op 487 / ~100× off peak / per-outer-cell task overhead" prose
(flagged in `FACT_CHECK.md`) with measured, checksum-gated numbers. Supporting artifacts:
`scaling-campaign-data/gap_profile.txt`, `gap_ceiling.csv`, `gap_decomposition.csv`.*

> **CORRECTION (2026-07-31, measured — see `MPQC_ABLATION.md`).** Two central claims below were
> reached by *source-reading* and are **refuted by directly profiling MPQC and ablating its features**:
> (1) "MPQC hits the same thread-starved skinny-GEMM kernel" — **false**: MPQC's cold residual is
> **BLAS-bound** (44–53% dgemm, ~3% sync) where the repro's is thread-starved (6–8% dgemm, ~60% sync).
> The single-node gap is in **how MPQC executes the residual** — not a shared ceiling, not the inner-tile
> layout, not the backend (MPQC-Pthreads single-node ≈ MPQC-PaRSEC). (A later pass, `MPQC_RUNTIME.md`,
> pins it to ~10× lower per-GEMM execution overhead / single-thread speed from MPQC's array
> construction — *not* the evaluator caller, which is concurrency-equivalent to the repro's static einsum.) (2) "PaRSEC is the one multi-node lever that matters" —
> **overstated**: MPQC-Pthreads scales ~6×/8 ranks, nearly as well as MPQC-PaRSEC (~1.1× apart), so the
> repro's poor scaling is *also* its static evaluator, not the Pthreads backend. The analysis below is
> retained for its ceiling/skinny-GEMM math (still valid), but its MPQC-parity and PaRSEC-lever
> conclusions are superseded by `MPQC_ABLATION.md`.

> **CORRECTION (2026-08-03).** Three more items below are now superseded by later measurement:
> (1) the pointer above to `MPQC_RUNTIME.md`'s "array-construction / per-GEMM overhead" is itself
> superseded — `MPQC_SINGLE_THREAD.md` refutes the array-construction/compaction lever (a no-op) and
> lands `SPTC_SCALE_GEMM` (the real 1-thread lever), and `MPQC_PROFILE_DEEP.md` FC1 measures the kernel
> **compute/dispatch-bound** (IPC 2.55, DRAM ≤23 % of peak) — so the "memory/latency-bound at the BLAS
> level" phrasing (Finding 2) should read "**latency/overhead-bound at low arithmetic intensity**", not
> memory-bound. (2) The single-rank component of the Finding-4 decomposition (repro np1 12.4/66/155 s)
> is the **gcc/MKL** era; the honest clang/OpenBLAS repro is 29.9 s @1thr (3.6×) and **beats MPQC at 8
> threads** — that term was toolchain + the now-fixed `fused_scale`, so only the *growing multi-node
> scaling term* survives. (3) "prototyped only … not landed in the full residual" is stale:
> `SPTC_SCALE_GEMM` was landed in the arena einsum path (1.53×@1thr), targeting op :487's broadcast
> scale. See `MPQC_SINGLE_THREAD.md` + `MPQC_PROFILE_DEEP.md`.

## Headline

The remaining repro-vs-MPQC gap is **cold time** (~4.5–5.5× at cc-pVTZ / np=16, `MPQC_COMPARISON.md`
§11). Profiling **corrects the prior story on three points** and reframes the gap:

1. The cold hotspot is **line 488** (the ToT×ToT contraction over the *outer* PAO index μ̃, forming
   `CSE37`), **not line 487** (the flat×ToT half-transform the docs blamed).
2. The cost is **not flops and not task/retile machinery** — it is **thread starvation on tiny
   fragmented tasks** (~70% of wall in `ConditionVariable::wait`/`sched_yield`, only 5.6% in `dgemm`;
   ~42% parallel efficiency on 8 threads).
3. Restructuring the hotspot into per-pair dense GEMMs has a **finite, capped ceiling of ~2.2–2.6×**
   — even a perfect hand-GEMM runs at only ~9–10% of peak because the GEMMs are **skinny** (per-pair
   PNO ≈ 60 vs contracted μ̃ ≈ 144–260). This skinny structure is **fundamental to the method as
   derived and is shared with MPQC** (both use the same non-proto-μ̃ SeQuant pipeline).

Decomposing the np=16 gap shows the per-op hotspot is only *half* of it (and the smaller half for
large molecules); the **dominant, growing** component is **multi-node distribution** of the giant
intermediate — where MPQC scales 7.5–11× over 16 ranks and the repro only 1.9–3.5×.

## Method

Single node (Xeon D-1548, 8 cores/16 threads, ~256 GFLOP/s double peak), owning-ToT, locked config
(`SPTC_COARSE_OCC=9 OCC_TILE=2 COARSE_PAD=0 TILES_PER_DIM=8`, `MAD_NUM_THREADS=8`), on the committed
cc-pVTZ leaves (`leaves/{C2H6,C3H8,C4H10}_coo`). Tools: TiledArray's built-in `TA_EINSUM_INSTRUMENT`
bucket profiler (`einsum/einsum_instrument.h`), `perf record -g`, a thread sweep, and a new
checksum-gated micro-benchmark `tools/gap_microbench.cpp` (TA `einsum` vs a hand per-pair BLAS++ GEMM
on identical data). Every comparison is validated against `ta_compute_checksum`.

## Finding 1 — root cause: thread-starved tiny tasks, not flops or machinery

`TA_EINSUM_INSTRUMENT` (aggregate over all cold einsums): **`local_kernel` 79–83%, `entry_fence`
17–21%, all machinery buckets (`Setup/Retile/CommSplitWorld/Harvest/Teardown`) ~0.** So it is not
per-tile task-*spawn*/retile overhead. The single biggest op is **`generated_t2_residual.cpp:488`**
(`… μ̃,Κ;a * … μ̃;a' -> … Κ;a,a'`, contracting the outer PAO index μ̃) — 30.5 s of 77.7 s einsum
time on C4H10; **line 487 does not appear in the top ops.**

`perf` on C2H6 cold shows what "`local_kernel`" wall actually is: **`madness::ConditionVariable::wait`
32% + `__sched_yield` 21% + scheduler/syscall ~15% ≈ 70% of wall is thread idle/sync**, `dgemm_kernel`
only **5.6%**. The thread sweep confirms it parallelizes but poorly — C2H6 cold T2 1/2/4/8 threads =
53.9/30.2/20.9/**15.9 s** = **3.4× on 8 threads (~42% efficiency)**. The ragged per-pair-PNO ToT
einsum fragments the μ̃-contraction into a huge number of tiny per-pair tasks; the MADNESS threadpool
starves and spends most of its wall synchronizing. (Data: `gap_profile.txt`.)

## Finding 2 — the ceiling from restructuring is ~2.2–2.6×, and it is capped

`gap_microbench` times the `CSE37` block (487→488, fenced) as TA `einsum` vs a hand-written per-pair
dense BLAS GEMM producing the identical result (checksums match to ~13 figures, identical nnz):

| molecule | TA einsum | hand-GEMM | ceiling | hand-GEMM GFLOP/s (% of peak) |
|---|---|---|---|---|
| C2H6  | 5.79 s  | 2.31 s  | 2.50× | 22.9 (9%) |
| C3H8  | 24.37 s | 9.40 s  | 2.59× | 26.2 (10%) |
| C4H10 | 73.36 s | 33.73 s | 2.18× | 21.8 (9%) |

Two things matter: the ceiling is **finite (~2.2–2.6×)** — so most of the `local_kernel` wall is
genuine GEMM work once you stop fragmenting it, consistent with removing the ~70% thread-sync — **and
even the hand-GEMM reaches only ~9–10% of peak.** These are **skinny GEMMs**: per-pair PNO width ≈ 60
against contracted μ̃ ≈ 144–260, so arithmetic intensity is low and the kernel is memory/latency-bound
*at the BLAS level regardless of who issues it*. You cannot approach peak by restructuring; the small
per-pair PNO dimension is intrinsic. (Data: `gap_ceiling.csv`.)

## Finding 3 — the skinny structure is fundamental and shared with MPQC

The skinny full-μ̃ half-transform is a property of the derivation, and **MPQC's residual has it too**:

- In the shared SeQuant CSV-CCSD derivation, the PAO index μ̃ and the aux index Κ are **global /
  non-proto** (`sequant-fork/.../convention.cpp:95-104`, `rules/df.cpp:63`, `rules/csv.cpp:31,81,92`);
  the per-pair (proto) mechanism attaches **only to the PNO index a**. So μ̃ is carried at full extent
  for every occupied pair by construction — the `(μ̃,Κ)`-per-pair form is structurally impossible.
- Real MPQC's CCSD residual uses the **same pipeline** and sends μ̃ to `approximate_size()` (full PAO),
  only PNO to `average_csv_extent` (`mpqc4/.../cck.ipp:1513-1540`). Its DF integral in the residual is
  a **global** `(Κ|pq)`, not a per-pair local-DF integral.
- MPQC *does* own DLPNO per-pair PAO domains (`pao_domains`, `g_tiles_from_ldf.h`), **but they live
  entirely in MP2/PNO construction and never enter the CCSD residual.** The only PAO-locality that
  survives into the residual is latent block-sparsity on C's contracted μ̃ leg, which at these scales
  covers ~the whole PAO space and does not de-skinny the GEMM.

A per-pair PAO domain in the residual would be an **approximation** (domain-truncation error), would
mainly reduce *work* for large molecules rather than raise intensity, and is **not expressible in the
current CSV pipeline** (would need a proto/domain on the PAO index plus per-pair LDF integrals). So
MPQC does **not** dodge this ceiling — it is fundamental to the method as derived.

## Finding 4 — the np=16 cold gap: half per-op, half (and growing) distribution

Decomposing the committed cold-T2 numbers (`gap_decomposition.csv`), the np=16 gap factors as
`single-rank per-op gap × multi-node scaling gap`:

| molecule | single-rank gap | scaling gap | = np16 gap | (MPQC scaling over 16 ranks) |
|---|---|---|---|---|
| C2H6  | 1.60× | 0.98× | 1.57× | 1.8× (too small to scale) |
| C3H8  | 2.12× | 2.11× | 4.47× | 4.6× |
| C4H10 | 2.02× | 2.35× | 4.75× | 7.5× |
| C5H12 | 1.68× | **3.25×** | 5.46× | **11.4×** |

- The **single-rank per-op gap (~1.6–2.1×)** matches the hotspot ceiling — it *is* the thread-starved
  skinny-GEMM hotspot, and a per-pair GEMM would recover it to ~MPQC single-rank parity (capped ~2×).
- The **multi-node scaling gap grows with size** (up to 3.25× for pentane): MPQC distributes the giant
  DF intermediate far better (7.5–11× / 16 ranks) than the repro (1.9–3.5×). This is **not the per-op
  kernel** — it is distribution — and for the largest molecule it is the *dominant* term.

## Conclusion — where the leverage is (and isn't)

- The "obvious" fix (per-pair GEMM for op 488) is real but **modest and capped (~2×)**: it removes the
  thread-starvation, recovering roughly the single-rank gap to MPQC parity, but it is skinny-GEMM
  limited (~10% peak) and does **nothing** for the growing multi-node scaling gap. Given the capped
  payoff, it was prototyped only to the checksum-validated kernel level (`gap_microbench`), not landed
  in the full residual.
- Because **MPQC hits the same per-op skinny ceiling**, MPQC's np=16 cold advantage is almost entirely
  **multi-node distribution** of the giant intermediate — the dominant, growing lever, and a
  multi-node/backend problem (not offline-addressable, not a kernel fix).
- Raising the *ceiling* at all requires a **method change** — a per-pair PAO domain in the residual
  (an approximation needing new SeQuant machinery + per-pair LDF integrals) or a genuinely different
  factorization. Neither is a tuning knob.

**Net:** the cold gap is not cheaply closable. ~2× of it is a recoverable (but capped) per-op kernel
issue; the rest — and the part that grows with molecule size — is multi-node distribution of an
intermediate whose per-op GEMMs are fundamentally skinny in this method, for MPQC as much as the repro.

## Corrections to prior prose (for `FACT_CHECK.md`)

- "op 487 is ~87% of cold T2" → the top cold op is **op 488** (ToT×ToT over outer μ̃); 487 is not in
  the top ops (owning-ToT). The 487-heavy reading was an arena-era / async-attribution artifact.
- "~100× off peak / per-outer-cell task overhead" → the hotspot is ~9–10% of peak as a *skinny GEMM*;
  the wall is **thread sync/starvation** (~70%), not task-spawn/retile machinery (those buckets ~0).
- The achievable restructuring speedup is **~2.2–2.6×, capped**, not order-of-magnitude.

## How MPQC drives einsum differently (same primitive)

A separate source investigation (over `mpqc4`, `sequant-fork`, the `cd53bd3` TiledArray) answers
"what does MPQC use *instead* of `TA::einsum`?" — **nothing at the primitive level.** Every
array×array contraction in MPQC's CSV-CCSD residual is `TA::einsum`: flat×flat (`result.hpp:381`),
flat×ToT (`:612`), ToT×ToT (`:621`/`:628`). No native `*`/SUMMA, no BTAS contraction, no hand GEMM.
The distribution map is byte-identical too (free-mode 2-D ProcGrid + `SlabbedPmap` replicating the
occupied-pair index, `cont_engine.h:824-948`; the legacy sub-world path is off in both; input pmap is
re-mapped away inside einsum — which is why the repro's `SPTC_CYCLIC_PMAP` did nothing). What differs
is how MPQC **drives and executes** the same einsum:

1. **Task backend — PaRSEC vs Pthreads (the one lever that matters; multi-node).** MPQC's SIF sets
   `MADNESS_TASK_BACKEND=PaRSEC` (`mpqc.def`, `mpqc.cpp:142`); the repro's TiledArray is Pthreads
   (`madness/config.h`). Same task graph, but PaRSEC's distributed scheduler overlaps comm/compute
   across nodes while MADNESS-Pthreads funnels every remote tile fetch through a single progress
   thread → the repro scales only 1.9–3.5× over 16 ranks while MPQC scales 7.5–11× (the dominant part
   of the np=16 cold gap in the decomposition above). Consistent with the earlier *single-node*
   finding that PaRSEC doesn't help (no network to hide): the backends diverge only where there's a
   network, i.e. multi-node. This was the source-level *prediction* for the load-bearing difference —
   **but the confirming experiment (E1) refuted it for the repro** (see below).

   **E1 result (REFUTED — 2026-07-30, `scaling-campaign-data/parsec_experiment.csv`).** Rebuilding the
   repro cold binary against the `cd53bd3-parsec` TiledArray (owning-ToT) and running it does **not**
   close the gap — it blows it up. PaRSEC-repro cold T2 is **4.6–64× SLOWER** than the Pthreads repro
   at every point, checksums rank-invariant: C2H6 np1 57.4 s vs 12.4 s (4.6×), C2H6 np2 174.8 s
   (negative scaling from np1), C4H10 np8 **743 s** vs Pthreads 64 s (11.6×) and MPQC 11.6 s (64×). The
   slowdown appears even **single-node (np1, no comm)** and only on **T2** (T1 is fine, 1.4 s), so it is
   PaRSEC's per-task scheduling overhead on the repro's **millions of tiny ToT tasks** — fine for
   small-basis with few tasks (the earlier §4 read of "PaRSEC ≈ Pthreads single-node") but catastrophic
   for cc-pVTZ's giant fragmented intermediate. **So the backend is not a lever the repro can flip.**
   MPQC gets *good* PaRSEC scaling on the *same* fine-grained ToT algebra, so its advantage must come
   from something the repro lacks that makes PaRSEC's distributed scheduling of those tiny tasks
   tractable (data-layout/pmap co-location, or a coarser effective task graph from the runtime
   evaluator) — an **open question**, but empirically it is *not* the `MADNESS_TASK_BACKEND` flag alone.

2. **Runtime cross-term cache dedup — real but wall-negligible.** MPQC walks a per-term binary forest
   through one shared `CacheManager` (`min_repeats=2`, keyed on the scalar-free tensor-network
   identity, `cache_manager.hpp`, `eval_node_compare.hpp`), so identical differently-scaled
   summand-root contractions collapse to one einsum. The repro's static per-term CSE can't merge
   summand roots. **Verified directly:** an exact-signature scan of `src/generated_t2_residual.cpp`
   finds **7** redundant two-operand einsums (208 total, 201 distinct, largest group 3×) — and **all
   of them are small o-space contractions; the giant μ̃Κ hotspot appears exactly once.** (A
   canonical/dummy-relabel-aware count — what the cache can in principle catch — is higher, ~29, but
   that over-counts via dummy relabeling + the repro's variable reuse, and still never touches the
   hotspot.) So this is a real mechanism but **negligible in wall time**, not a meaningful lever.

3. **Sparse screening threshold — minor.** MPQC runs the residual under `csv:tTA` = 1e-8
   (`cck.ipp:958`); the repro uses TA's default (~1.19e-7, overridable via `SPTC_SPARSE_THRESHOLD`).
   A work difference in principle, but the repro's residual was earlier found threshold-invariant
   (1e-8…1e-16 give the same checksum), so the wall impact is expected to be small. Testable.

**Ruled out (not repro disadvantages):** cross-iteration persistence (the repro's warm/cold split is
compute-equivalent to MPQC's P/NP cache — same t-independent work done once); ReorderSum
(liveness/locality only, same flops); the aux-Κ batched evaluator (memory-only, default-off, and on
the residual not the energies); R2 symmetrization (MPQC does slightly *more* per iteration — one
O(o²v²) permute/add the repro correctly skips).

**Bottom line (revised after E1).** MPQC uses the same `TA::einsum`, the same distribution algorithm,
and (once cold) the same t-independent work; the cross-term cache and threshold are real but
wall-negligible. The obvious candidate for the multi-node lever — the PaRSEC task backend — was
**tested and refuted**: the repro on PaRSEC is 4.6–64× *slower*, because PaRSEC's per-task overhead is
pathological for the repro's millions of tiny ToT tasks (MADNESS-Pthreads handles fine-grained
shared-memory tasks far better here). So the scaling gap is **not** a backend flag. MPQC scales well
with PaRSEC on the same algebra, so its multi-node advantage lives in how its work is *laid out /
scheduled* to make those tiny tasks tractable for a distributed runtime — layout/pmap co-location or a
coarser effective task graph — which the repro, emitting a flat static einsum sequence over a
default-pmap giant intermediate, does not reproduce. **That, not the einsum primitive or the backend,
is the real open lever** — and it is a substantial layout/scheduling change, not a knob.
