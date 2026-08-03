# Deep re-profile: hardware counters, thread-scaling, warm, and cross-node comm

*2026-08-02, node3 + cluster node16-31 (both Xeon D-1548, 8 cores, 12 MiB L3, 1 NUMA node), `perf 5.15`,
`build-prof` = arena `-O2 -g -fno-omit-frame-pointer`. This is the "deep re-profile for new levers" pass:
every prior profile was flat perf self-time + wall time; this one adds hardware counters, DRAM bandwidth,
per-thread-count call-graph attribution, a warm profile, and a real-np comm/compute decomposition. Data:
`scaling-campaign-data/{hwcounters,thread_sweep,warm_profile,comm_profile}.csv` +
`scaling-campaign-data/profiles/`.*

## TL;DR / verdict

**No new knob-lever exists — but the profile CORRECTS one wrong claim and UNIFIES all three regimes'
inefficiency under the single lever already identified.** Specifically:

1. **The "memory-bound" claim is refuted, measured.** The single-thread μ̃Κ half-transform runs at **IPC
   2.55** and pulls only **~1.4-3.1 GB/s DRAM (≈4-9 % of the D-1548's ~34 GB/s peak)** with L1-resident
   data — it is **compute/instruction-dispatch-bound**, not memory-bound. `MPQC_SINGLE_THREAD.md`'s
   "re-streams C 19.7 B times → memory-bound" inference was wrong; the C tile stays in L1/L2.
2. **The bottleneck in every regime is the same thing: fine-grained tensor-of-tensor tasks that don't keep
   BLAS fed.** It manifests as instruction-dispatch overhead at 1 thread, thread-starvation at 8 threads
   (cold *and* warm), and non-distributing replicated compute at multi-rank. This is MPQC's
   runtime-evaluator work-coalescing + solver-inherited layout — the generator/backend project, now with
   hard, unified evidence across all three regimes.
3. **Every knob is refuted by direct measurement** (memory-locality: no memory wall; wait-policy: yield
   already optimal, busy slower).

## P1 — Hardware counters: memory-bound REFUTED (`hwcounters.csv`)

C2H6, 1 thread, `build-prof`, uncore-IMC DRAM bandwidth:

| | wall | IPC | instr | L1-dmiss | DRAM BW (T2) | % of peak |
|---|---|---|---|---|---|---|
| baseline | 31.06 s | **2.55** | 274.9 B | 4.3 % | 1.4-3.1 GB/s | **4-9 %** |
| scale-GEMM | 19.34 s (**1.60×**) | 1.95 | 155.9 B | 6.2 % | — | — |

IPC 2.55 (a memory-bound kernel is < 1.0), L1-miss 4.3 %, LLC traffic ~5 M/s (≈0.3 GB/s), DRAM at 4-9 % of
peak — the kernel is **compute-bound**, and scale-GEMM wins purely by **cutting instruction count**
(275 B → 156 B; the 358 M per-cell scalar AXPYs collapse into GEMMs), not by improving locality. There is
no memory-locality lever because there is no memory wall.

## P2 — Per-thread-count self-time: starvation grows; wait-policy refuted (`thread_sweep.csv`)

C2H6 cold T2, flat self-time as `MAD_NUM_THREADS` goes 1→8 (first call-graph attribution per thread count):

| symbol | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| `fused_scale` (compute) | 34 % | 26 % | 17 % | 11 % |
| `ConditionVariable::wait` (idle) | — | — | 26 % | **45 %** |
| yield syscalls | — | 14 % | 12 % | 11 % |
| `dgemm` | 9 % | 7 % | 5 % | 4 % |

Compute parallelizes and shrinks; **thread-starvation grows with thread count** — at 8 threads ~45 % idle
+ ~11 % yield-syscall = ~56 % overhead, only ~15 % real compute (wall scales 4.49×/8 = 56 % efficiency).
The threadpool cannot keep 8 threads fed with the fine-grained ToT tasks. **Wait-policy A/B (N=8):** yield
6.4 s < sleep 6.7 s < busy 6.8 s — **yield is already optimal, busy is slower**, so this is *not* a policy
knob; the only fix is coarser tasks (task-coalescing), bounded by the dense-intermediate-blowup constraint.

## P3 — Warm profile: also dispatch-bound, not BLAS-bound (`warm_profile.csv`)

C2H6 warm (`ta_warm_t2`, t-dependent block only, DF/CSV precomputed), 8 threads, wall 1.65 s: `syscall`
17.5 % + `ConditionVariable::wait` 14.6 % + `entry_SYSCALL` 4.3 % = **~36 % thread-sync**, real compute
(fused_scale 7.9 % + dgemm 7.2 %) only ~15 %. So even the steady-state warm iteration is
**starvation/dispatch-bound**, unlike MPQC's warm (dgemm 24 %). The fine-grained-task problem is *not* a
cold-construction artifact — it persists into the warm loop, the real CCSD-iteration cost.

## P4 — Cross-node comm vs compute: scaling term is non-distributing COMPUTE (`comm_profile.csv`)

C3H8 cold, 1 rank/node × 8 threads, `TA_EINSUM_INSTRUMENT`, **real np on actual nodes** (prior runs were
oversubscribed on one node):

| np | einsum/rank | local_kernel (compute)/rank | entry_fence (sync) |
|---|---|---|---|
| 4 | 25.0 s | **20.7 s** (83 %) | 17 % |
| 8 | 19.6 s | 16.0 s (81 %) | 18 % |
| 16 | 17.7 s | 14.1 s (80 %) | 20 % |

Going 4→16 ranks (**4× more nodes**), per-rank compute drops only **20.7→14.1 s = 1.47×**, not 4× (37 % of
ideal strong-scaling). The dominant μ̃Κ half-transform is **replicated across ranks, not divided**; sync
(`entry_fence`) is a flat ~18 % and network buckets (`commsplit`/`contract+fence`) are ~0. So the
multi-rank scaling term is **non-distributing local compute, not communication** — confirming
`MPQC_MULTIRANK.md` §5b (einsum's synthesized ProcGrid/CyclicPmap scatters the frozen-core zeros so surplus
procs land on the replicated slab axis) at real np. The lever is work distribution / solver-inherited
layout upstream of `TA::einsum` — the generator/backend project, not a knob (einsum discards operand pmap).

## The unified picture

One mechanism, three faces:

| regime | symptom | measured here |
|---|---|---|
| 1 thread | too many cheap AXPY+dispatch instructions | IPC 2.55, 275 B instr, DRAM 4-9 % of peak (P1) |
| 8 threads (cold & warm) | pool starves on fine-grained ToT tasks | 45 %/36 % idle-wait, 56 %/85 % non-compute (P2/P3) |
| multi-rank | dominant op replicated, not divided | per-rank compute 1.47× for 4× ranks (P4) |

All three are the same underlying thing — **TiledArray's fine-grained ToT tasks never keep the BLAS units
fed**, which MPQC solves with its runtime-evaluator work-coalescing (44-53 % dgemm self-time) + its
solver-inherited balanced layout. scale-GEMM already fixed the 1-thread face (fewer, bigger GEMMs). The
8-thread and multi-rank faces need the *scheduling/layout* half of that solution, which is the generator/
backend project — not reachable by any env knob (all refuted).

## Method notes (for the next profiling pass)

- `build-prof` (arena `-O2 -g -fno-omit-frame-pointer`) is the profiling binary; the git-`build/` dir has an
  **empty `CMAKE_BUILD_TYPE` (unoptimized, ~5× slow)** — never profile with it. Stage leaves to `/local`
  (NFS load otherwise inflates to ~28 s). DRAM BW via `perf stat -a -e uncore_imc_{0,1}/cas_count_{read,write}/ -I`.
- `fp` call-graph unwinding breaks through OpenBLAS's frame-pointer-less assembly (a 54 % `[unknown]` frame
  in flame graphs) — **flat `--no-children` self-time is reliable; call-graph flame graphs are not** without
  a `--call-graph dwarf` rebuild of OpenBLAS. `stalled-cycles-frontend/backend` are `<not supported>` on
  Broadwell-DE; `LLC-load-misses` reads 0 (non-counting) — use IMC CAS for the memory verdict.
- **perf is installed only on node3, not node16-31** (and `perf_event_paranoid` varies, e.g. node17=4). A
  per-rank perf self-time on the cluster needs perf pushed to all nodes + paranoid set — deferred; the
  `TA_EINSUM_INSTRUMENT` bucket decomposition already answered comm-vs-compute without it.

## Verdict on "is there a new lever?"

**No.** The deep profile confirms the characterized set is complete and refutes the last inferred-but-
unmeasured claim (memory-bound). The one lever that would move the 8-thread and multi-rank numbers is
task-coalescing / keeping BLAS fed — MPQC's runtime evaluator + solver-inherited layout — which is a
generator/backend project already scoped in `MPQC_MULTIRANK.md`, not an env knob. Any implementation effort
should target that (fenceless dataflow chaining + up-front balanced layout upstream of einsum), with the
honest expectation, per `MPQC_RUNTIME_EVAL.md`, that a naive evaluator port alone does not capture it.

---

## Fact-check (2026-08-03): adversarial, cross-molecule, both-sides, cluster-confirmed

The P1-P4 verdict above came mostly from **C2H6 (smallest molecule), single-node, mostly 1 thread**. This
pass re-tests each conclusion on the **largest molecules (C4H10, C5H12) and worst-case configs**, fills the
gaps (MPQC-side counters, cluster per-rank perf, 8-thread bandwidth), and states CONFIRMED/REVISED per
conclusion. Data appended to the CSVs; new flame graph `profiles/flame_C4H10_t8.svg`.

**Method corrections surfaced by fact-checking my own runs (worth recording):** (1) the **arena binary
crashes on C4H10/C5H12** (giant-intermediate blowup) — the big molecules need the **owning** binary
(`$W/bin`, arena-sym-free); a first FC1 pass silently profiled only the LOAD phase. (2) DRAM bandwidth must
be **T2-isolated** — the memory-heavy COO *load* alone hits 87 % of peak and produced a false "memory-
bound" alarm; isolating the last `wall_s` seconds of the perf `-I` timeline fixes it. (3) `/local`, `/tmp`,
`/users` are one root disk here — big-molecule staging + a 786 M dwarf perf.data filled it (MPI_Init then
fails with PMIX OOM); dwarf call-graph capture is impractical on the long 8-thread runs, `fp` flat
self-time is the reliable readout.

### FC1 — memory-bound: **CONFIRMED** (adversarially). `hwcounters.csv`
T2-isolated counters, owning binary:

| (T2 only) | IPC | DRAM avg | DRAM peak |
|---|---|---|---|
| C4H10 1thr | 1.80 | 2.7 GB/s (8 %) | 8.7 (25 %) |
| C4H10 8thr | 0.90 | 8.0 GB/s (23 %) | 17.3 (51 %) |
| C5H12 8thr | 0.87 | 7.1 GB/s (21 %) | 20.4 (60 %) |

Even the worst case (biggest DF + 8 threads) sustains only **21-23 % of the ~34 GB/s peak** (bursts to
~55 %, never saturated). The low 8-thread IPC (~0.9) is **thread-starvation** (FC4a: 46.7 % condvar-wait),
not memory stalls — a memory-bound kernel would show low IPC *and* saturated DRAM; here DRAM is far from
saturated. Upgrades P1 from 1-thread-C2H6 to cross-molecule + 8-thread.

### FC2 — thread-starvation: **CONFIRMED** (structural, not a small-molecule artifact). `thread_sweep.csv`
C4H10 8thr cold self-time: `ConditionVariable::wait` 24 % + `std::_Function_handler` (per-cell dispatch)
19.5 % + syscall 9.2 %, `dgemm` only 6.5 %. On the bigger molecule the mix shifts from *pure wait* (C2H6
45 %) toward *per-cell dispatch* (19.5 %), but it is still ~55 % sync+dispatch and **not BLAS-bound** — the
starvation/dispatch bottleneck is structural across molecule size.

### FC3 — warm also dispatch-bound: **CONFIRMED** at larger size. `warm_profile.csv`
C3H8 warm: ~40 % syscall/sync, `dgemm` negligible. C4H10 warm: condvar 18.5 % + dispatch 15 % + syscall
10.7 %, `dgemm` 6.9 %. Both remain sync/dispatch-dominated (not BLAS-bound like MPQC's warm).

### FC4 — multi-rank = non-distributing COMPUTE not comm: **CONFIRMED** (both parts). `comm_profile.csv`
(a) Per-rank `perf record` self-time (np8 C3H8, real cluster): rank0 = `ConditionVariable::wait` 46.7 % +
syscalls ~17 % — **zero MPI/progress/comm functions in the top**. The per-rank bottleneck is intra-rank
thread-starvation, not communication. (b) Non-distribution reproduces on C4H10 (the ~7×-gap molecule):
per-rank `local_kernel` 36.2 → 24.7 → 17.3 s for np 4/8/16 = **2.08× for 4× ranks** (52 % of ideal),
balanced across ranks.

### FC5 — MPQC-side counters: **CONFIRMED** (both sides compute-bound; edge is BLAS-feeding). `mpqc_counters.csv`
MPQC ethane under `perf stat -a`: the CCSD-residual phase runs at IPC ~1.1-1.3 and DRAM **2-6.7 GB/s
(~6-20 % of peak)** — MPQC's residual is **also compute-bound, not memory-bound**, like the repro. (An
earlier high-IPC 1.8 / near-zero-DRAM phase is the cache-resident SCF/equation-generation.) MPQC does the
55-term T2 in **2.1-3.1 s vs the repro's ~7 s** at 8 threads, with 44-53 % dgemm self-time (ABLATION). So
**neither side is memory-bound**; MPQC's ~2.5-3× single-node edge is entirely *keeping BLAS fed* (no
thread-starvation), not memory bandwidth and not a different kernel. Caveat: the residual is brief (2-3 s),
so 1 s-interval isolation is approximate — the qualitative "not memory-bound" is clear, the exact IPC less so.

## Fact-check verdict

**All five conclusions CONFIRMED; none revised.** The deep-profile verdict is upgraded from
"single-molecule (C2H6), single-node, mostly 1-thread inference" to **cross-molecule (through C5H12),
8-thread, warm, real-cluster, and both-sides counter-measured**:
- memory-bound is refuted even adversarially (biggest DF + 8 threads → DRAM ≤23 % of peak); **both** repro
  and MPQC are compute-bound, so the gap is not memory on either side;
- the one bottleneck — fine-grained ToT tasks that don't keep BLAS fed — is structural across molecule size
  (its 8-thread mix shifts from pure thread-wait toward per-cell dispatch as molecules grow, but never
  becomes BLAS-bound), persists into the warm loop, and at multi-rank is non-distributing compute (not
  comm: zero MPI in the per-rank self-time);
- no new knob-lever; the lever remains MPQC's runtime-evaluator work-coalescing + solver-inherited layout
  (the generator/backend project), now counter/comm/cross-molecule-confirmed.
