# Contraction IR

The generated C++ sequence is executable but does not expose enough structure
for planning and cross-framework analysis. Contraction IR (CTIR) is the
human-readable, value-oriented view of the same SeQuant evaluation forest.

Each operation records stable operation and value identifiers; input/output
annotations and tensor families; binary contraction indices and coefficient;
flat versus ragged tensor-of-tensor storage; volatile-amplitude dependence;
persistence/reuse eligibility; estimated work, communication, and intermediate
size; and batching and layout constraints.

CTIR preserves occurrence identity. Algebraically equal residual terms are not
collapsed, and R1, R2, and R0/energy occurrences remain distinct. This is the
same rule used by the authoritative 81-term native residual catalog.

## Why einsum text is insufficient

A flat einsum describes index incidence but not the CSV/PNO ragged inner
domains, block-sparse outer support, physical tile layout, process map, or
whether an intermediate is persistent across residual replays. Those facts
change memory and communication cost without changing the printed equation.

CTIR therefore treats tensor-of-tensor metadata as part of the operation
contract. A backend may lower an operation to native ragged tensors, explicit
per-pair contractions, or another faithful representation, but it must not
silently pad or merge occurrences.

## Generation and validation

CTIR is emitted from the same copied SeQuant forest as the TiledArray and NumPy
generators. A valid export must preserve the 26 R1 and 55 R2 occurrence IDs,
every binary dependency and result annotation, enough leaf metadata to bind
the same COO inputs, and the source/generator fingerprint. Its lowering must
reproduce the residual checksums.

Generated CTIR is a run artifact rather than maintained documentation. Keep it
beside the input/profile that produced it; Git history contains previous
campaign samples.
