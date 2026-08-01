# runtime-eval SeQuant-side diagnostics

Modifications applied to the installed SeQuant header
`SeQuant/core/eval/backends/tiledarray/result.hpp` (install prefix
`/users/jianjian/sequant-fork/install-eval`) while porting the runtime evaluator
(`src/ta_runtime_eval_main.cpp`, see `docs/MPQC_RUNTIME_EVAL.md`). The install is on
ephemeral disk; `result.hpp.modified` is the full modified file so the two changes
survive re-provisioning. Neither is a fix for the evaluator crash — both are kept
as diagnostics / defensible-direction changes.

## Change 1 — flat-first operand order in `ToT * T -> ToT`

`ResultTensorOfTensorTA::prod`, the `other.is<that_type>()` branch. Upstream emits
`TA::einsum(this_ToT, other_flat, …)` (ToT operand first). This TiledArray fork's
de-nesting einsum is not operand-order-symmetric there: ToT-first SIGSEGVs in
`SparseShape::gemm`; flat-first (the order the static generator always emits,
`generated_t2_residual.cpp:85`) gets further before heap-corrupting. Swapped to
flat-first (value-identical — operand listing is commutative).

## Change 2 — `SPTC_PROD_TRACE` env-gated annotation prints

Two `std::fprintf(stderr, …)` in `ResultTensorTA::prod` (flat×flat) and
`ResultTensorOfTensorTA::prod` (ToT branch) printing `lannot / rannot / this_annot`
+ operand types before each `TA::einsum`. Enabled by setting `SPTC_PROD_TRACE=1`.
Used to prove the evaluator's crashing-step annotations are byte-identical to a
hand-driven replay that runs cleanly (`docs/MPQC_RUNTIME_EVAL.md`).
