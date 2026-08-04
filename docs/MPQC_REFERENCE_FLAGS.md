# MPQC as a cross-check reference — existing runtime flags, no rebuild

This repo *is* the controllable, cache-free, explicit-sequence benchmark. MPQC is only a
**cross-check reference**: a way to get known-correct numbers and to see the sequence MPQC's runtime
evaluator actually executes. Everything you need for that already exists as **runtime KeyVal flags on the
current instrumented SIF** — no MPQC/SeQuant source change and **no Singularity rebuild**.

This document exists so nobody re-derives (or re-implements) inside MPQC what is already a runtime toggle.

> **Scope boundary.** These flags let you *observe* and *turn off caching in* MPQC's evaluation. They do
> **not** let you *control* (reorder / substitute / prune) the sequence MPQC executes — that lives in
> SeQuant's `optimize`/`binarize` and would require editing the source and rebuilding the SIF
> (`mpqc-benchmark/bin/mpqc.def`, `sudo apptainer build`, ~45–60 min, ~2.9 GB, passwordless root,
> single-rank only). Controllability is exactly what this repo owns instead. See
> `docs/HARNESS_VS_EXPERIMENTS.md` and `README.md`.

All KeyVal keys below go in the `wfn` (`type: CCk`) block of the run JSON. Reference config:
`../mpqc-benchmark/traces/checksum-run/ethane-checksum-instrumented.json`.

---

## 1. Cache-free run (no cross-node reuse) — zero code

MPQC's intermediate reuse is the `CacheManager` (memoizes intermediates repeating ≥2×, split
persistent/volatile) plus `ReorderSum` (aligns summands so common sub-sums coincide). Both are switched
by one flag:

| Key | Default | Set to | Effect |
| --- | --- | --- | --- |
| `cache_imeds` | `true` | `false` | `cache_manager()` returns the empty manager, `optimize` runs `NoReorder`, and `evaluate` is called **without** a cache. This is MPQC's cache-free mode. |
| `seq_opt` | `true` | `false` | *(optional)* also skips `sequant::optimize` entirely (no single-term factorization/binarization tuning). |

```jsonc
"wfn": {
  "type": "CCk",
  "method": "df",
  "seq_opt": "false",       // optional — also skip optimize()
  "cache_imeds": "false",   // <-- cache-free
  ...
}
```

Source: `mpqc4/src/mpqc/chemistry/qc/lcao/cc/cck.ipp` — `cache_imeds` parsed at `:196`, `seq_opt` at
`:174`; the empty-manager / NoReorder / no-cache branches at `cck.h:621`, `cck.ipp:1168,1527,1237-1238,1690`.

**Caveat (honest):** even cache-free, each term is still binarized, so *within-term* intermediates exist by
construction — nothing is *retained across nodes/terms/iterations*, which is what "cache-free" means here.
The truly zero-reuse, fully-expanded spec is the trace's `Term | Begin` line (§2), not the live run.

## 2. Dump the executed contraction sequence — zero code

The SIF is built with `-DSEQUANT_EVAL_TRACE=ON` (`mpqc-benchmark/bin/mpqc.def:154`), so tracing is
compiled in and **gated at runtime**:

```jsonc
"wfn": {
  "type": "CCk",
  ...
  "sequant": { "trace": { "eval": "true", "eval_level": "1" } }
}
```

This makes `sequant::evaluate` emit, per binary node,
`Eval | <Product|Sum|Tensor> | <t>ns | left=…|right=…|result=… | checksum=nnz,sum,sumsq,max_abs | <label>`
and per term a `Term | Begin | <coef> <fully-expanded-expr>` line
(formatters in `sequant-fork/SeQuant/core/eval/eval.hpp:269-340,619-664`). Live example:
`../mpqc-benchmark/traces/checksum-run/ethane-checksum-v2.log`.

**The parse/replay toolchain already exists** in `../mpqc-benchmark/traces/`:
- `parse_trace.py` — log → `steps.csv` (one row per eval step).
- `term_begin_mapper.py` — parses the `Term | Begin` fully-expanded trees.
- `extract_trace_equations.py` → `*.extracted_equations.txt` — staged `eqN: LHS * RHS -> RESULT` blocks
  with a leaf→`.tns` mapping. Its own header states it is built from each real `Term|Begin` line's
  **fully-expanded, NON-deduplicated (zero-CSE, no-reuse)** expression — i.e. exactly the framework-
  agnostic no-reuse sequence this benchmark drives. See
  `traces/checksum-run/ethane-checksum-v2.extracted_equations.txt`.

**Overhead:** each traced step walks every element to checksum it, so use the **trace-off** config
(`ethane-perf.json`) for timing and the **trace-on** config (`ethane-checksum-instrumented.json`) for
correctness/sequence dumps. A low-overhead structure-only trace *would* need a source edit + rebuild —
out of scope (§Scope boundary).

## 3. Dump leaf data for external replay — zero code

To feed the same leaves to another framework (e.g. `tools/numpy_runner.py` here), MPQC can write the COO
leaf tensors used by a named expression:

```jsonc
"sequant": { "trace": { "selected": { "write_tns": "true", "tns_mode": "both" } } }
```

`evaluate_traced_exprs` (`cck.ipp:1850+`) writes `.tns` COO leaves (`tns_mode` = `tile`|`element`|`both`).
Existing dumps live under `/proj/perf-model-gpu-PG0/jianjian-alkanes-v3/tns/` and the repro reads this exact
COO format via `src/coo_loader.h` / `src/ta_tensor_loader.h` — so the same `.tns` leaves drive **both** the
TiledArray harness and the NumPy backend, making cross-framework checksums directly comparable.

---

## What this buys the benchmark

- **Known-correct reference numbers** at any cache setting, with no rebuild: run the existing SIF with
  `cache_imeds=false` (§1).
- **The exact no-reuse sequence MPQC runs**, as a text spec: `sequant:trace:eval` +
  `extract_trace_equations.py` (§2) — the same shape this repo emits from SeQuant with `SPTC_NO_CSE=1`.
- **The same leaf data**, in the COO format the repo already loads (§3).

## What it does NOT buy — and why the work belongs here

MPQC cannot cheaply give a **controllable** sequence (reorder/substitute/prune from outside) or run a
**different** contraction order than its own `optimize`/`binarize` chose. Doing that means editing SeQuant
source inside MPQC and rebuilding the SIF (~45–60 min, root-only, single-rank). This benchmark exists
precisely to provide that control in minutes, against stock backends and any future framework — so
modifying MPQC "for the purpose" is redundant. Keep MPQC as the reference; keep the controllable,
cache-free, multi-backend experimentation here.
