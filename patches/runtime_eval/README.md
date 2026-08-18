# runtime-eval SeQuant-side diagnostics

> **EXPERIMENT — NOT part of the benchmark harness.** The `sequant::evaluate()` runtime-evaluator
> driver (`src/ta_runtime_eval_main.cpp`) reproduces MPQC's runtime tree-walk as a *control*; it is
> CMake-gated by `SPTC_BUILD_RUNTIME_EVAL=ON` and **default-off**. It builds a second tensor-evaluation
> path (the opposite of this repo's "explicit, controllable sequence" goal) and is measured to be ~1.2×
> slower — kept for comparison, not as the benchmark. See `docs/HARNESS_VS_EXPERIMENTS.md`.

`result.hpp.modified` is the full modified copy of the installed SeQuant header
`SeQuant/core/eval/backends/tiledarray/result.hpp` (install prefix
`/users/jianjian/sequant-fork/install-eval`), used while porting the runtime
evaluator (`src/ta_runtime_eval_main.cpp`). The
install is on ephemeral disk; this copy preserves the changes.

The only change vs upstream is an **`SPTC_PROD_TRACE`-gated diagnostic** — a few
`std::cerr`/`fprintf` in `ResultTensorTA::prod` (flat×flat) and
`ResultTensorOfTensorTA::prod` (ToT branch) printing `lannot / rannot /
this_annot` + operand types + operand `trange()`s before each `TA::einsum`.
Enabled with `SPTC_PROD_TRACE=1`. This was the tool that localised the crash: it
printed the crashing `ToT * T` einsum's ToT operand as **rank-2 outer (9,144)**
under a **rank-3 annotation** — revealing the driver's leaf yielder was serving
`c1_tot` (singles coeff) where `c2_tot` (doubles) was required. No behavioural
change to the evaluator when the env var is unset.

(An earlier experiment also swapped the `ToT*T` einsum to flat-first; it was
**reverted** — the crash was the leaf mis-mapping, not operand order, and
ToT-first works fine once the yielder is correct.)
