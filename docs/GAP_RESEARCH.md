# Cold-time gap: root cause, ceiling, and where the leverage really is

*Offline research (2026-07-30, node3 single-rank unless noted). Replaces the previously
uncommitted "~87% is op 487 / ~100× off peak / per-outer-cell task overhead" prose
(flagged in `FACT_CHECK.md`) with measured, checksum-gated numbers. Supporting artifacts:
`scaling-campaign-data/gap_profile.txt`, `gap_ceiling.csv`, `gap_decomposition.csv`.*

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
