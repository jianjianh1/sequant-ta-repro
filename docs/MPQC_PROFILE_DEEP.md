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
   2.55** and pulls only **~1.4-3.1 GB/s DRAM (≈5-9 % of the D-1548's ~34 GB/s peak)** with L1-resident
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
| baseline | 31.06 s | **2.55** | 274.9 B | 4.3 % | 1.4-3.1 GB/s | **5-9 %** |
| scale-GEMM | 19.34 s (**1.60×**) | 1.95 | 155.9 B | 6.2 % | — | — |

IPC 2.55 (a memory-bound kernel is < 1.0), L1-miss 4.3 %, LLC traffic ~5 M/s (≈0.3 GB/s), DRAM at 5-9 % of
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
| 1 thread | too many cheap AXPY+dispatch instructions | IPC 2.55, 275 B instr, DRAM 5-9 % of peak (P1) |
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
