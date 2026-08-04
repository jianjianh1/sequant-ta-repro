# backends/numpy — the NumPy einsum backend (second framework)

`generated_t{1,2}_residual.py` are the CSV-CCSD residual **cache-free** contraction sequences emitted for
**NumPy einsum**, from the *same* SeQuant forest as `src/generated_t{1,2}_residual.cpp` (TiledArray). They
are produced by the additive `NumPyEinsumGenerator` block in the derivation tool
(`sequant-fork/tests/manual/test_csv_ccsd_derivation.cpp`) with `SPTC_NO_CSE=1` — proving that targeting a
second framework is a small generator block, not a rewrite.

Run with `../../tools/numpy_runner.py` (converts the `.tns` leaves to dense `.npy`, binds extents, execs the
program, checksums the result like the TA path).

**Status / caveat:** the `np.einsum` *subscripts* are faithful to the same sequence, but the stock generator
tags per index *space*, so this CSV residual is not yet runnable end-to-end — μ̃/Κ collide on the tag byte
`0xCE` (these files are therefore not valid UTF-8; read as latin-1), the two PNO extents collapse to one
`dim_a`, and the tensor-of-tensor `C` leaves emit with too few tags to map to a `.tns`. Making it run needs
per-index tags/extents + ToT lowering in the generator. Full detail and the path forward:
`../../docs/NUMPY_BACKEND.md`.
