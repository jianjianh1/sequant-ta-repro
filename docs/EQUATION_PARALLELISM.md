# Equation parallelism

The native cache-free residual exposes 81 independent top-level occurrences:
26 R1 and 55 R2. They may execute concurrently only after their shared leaf
inputs are ready, and their outputs must be accumulated with a documented
ordering policy.

Equation-level scheduling is distinct from TiledArray's parallelism inside one
binary contraction. A scheduler must account for critical-path time, available
contraction parallelism, peak simultaneous intermediate memory, communication,
process-map compatibility, volatile versus persistent work, and deterministic
accumulation.

## Safe experiment contract

1. Keep the occurrence population and selected binary trees fixed.
2. Use the serial equation schedule as the correctness baseline.
3. Bound concurrency by measured or conservatively estimated peak memory.
4. Fence only at declared dependency and measurement boundaries.
5. Compare full residual checksums and energy, not timing rows alone.
6. Report rank placement and MADNESS threads; equation concurrency can
   oversubscribe the workers used inside each contraction.

R0/energy operations in the older 86-ID diagnostic catalog are not residual
work and must not be added to this scheduler's 81-term population.
