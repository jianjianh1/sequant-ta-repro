# Fair comparison with MPQC

Use this repository for the explicit SeQuant → TiledArray sequence and the
sibling `mpqc-benchmark` repository for authoritative native MPQC measurements.
Do not compare against the retired trace-derived or isolated-panel artifacts.

## Contract

A valid comparison must use:

- the same molecule, basis, converged state, and exported leaf tensors;
- `cache_imeds=false` on MPQC and the `SPTC_NO_CSE=1` generated sequence here;
- the same pinned TiledArray lineage, compiler family, BLAS, rank placement,
  thread count, and CPU affinity;
- fresh processes or an explicitly documented warmup policy;
- the owning-ToT canonical build in this repository;
- all quarantined experiment gates disabled.

Gate correctness on stable residual structure and gauge-invariant checksums.
Keep signed sums and extrema as diagnostics when orbital gauge or floating
reassociation can change them. Report process-lifetime peak RSS separately
from contraction timing.

## Authoritative workload

The native no-cache residual is 81 occurrence-preserving terms: 26 R1 and 55
R2. MPQC's older 86-ID trace catalog also included five R0/energy occurrences
and used weaker matching; it is diagnostic only. The old 16-operation isolated
panel was selected by harness tooling rather than the complete native workload
and must not be used for a headline comparison.

For current collection and validation commands, use
`../mpqc-benchmark/README.md` and its `native-residual-bench.py` and
`run-native-equation-sweep.py` workflows. This repository intentionally keeps
no duplicate performance-result ledger: generated campaign data belongs in a
run directory, and historical claims remain in Git history.
