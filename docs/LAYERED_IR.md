# Layered residual IR

One representation cannot simultaneously preserve chemistry semantics and
describe executable distributed tensor operations. Model the residual as this
lowering stack:

1. **Equation IR** — residual targets, coefficients, tensor labels, index
   spaces, symmetry, and restrictions; all occurrences remain distinct.
2. **Term DAG** — products, sums, volatile-amplitude dependence, and explicit
   common-subexpression/replay policy.
3. **Binary contraction IR** — a selected tree for every product plus
   intermediate annotations. This is the level qblock and SeQuant planners
   change.
4. **Storage IR** — flat or ragged tensor-of-tensor layout, tiles, sparsity
   shape, ownership, and process map.
5. **Backend program** — concrete TiledArray calls, or another framework's
   faithful lowering.

Each lowering must be deterministic and fingerprinted. A transformation may
change only the properties owned by its layer: a tree-selection pass may
change binary grouping but not term population or leaf semantics.

Validation boundaries preserve 26 R1 and 55 R2 occurrences, algebraic value
and reuse policy, ragged inner domains and outer support, and finally the full
residual checksums with documented floating reassociation.

This separation prevents treating the 86-ID trace catalog as an 81-term
residual, treating deduplicated families as occurrences, or describing a
padded flat tensor as equivalent to a ragged CSV/PNO tensor-of-tensors.

See `CONTRACTION_IR.md` for the value-level representation and
`HARNESS_VS_EXPERIMENTS.md` for the boundary between a faithful lowering and a
backend experiment.
