# NumPy einsum backend — a second framework from the same forest

**Goal it serves.** "Benchmark the same contraction sequence against any future framework." This documents
standing up **NumPy einsum** as a second backend, emitted from the *same* SeQuant `EvalNode` forest as the
TiledArray sequence — proving that adding a framework is a small additive generator block, not a rewrite —
and it records, honestly, exactly what the stock generator still needs to make the CSV-CCSD program *run*.

## What works: the export mechanism (proven)

SeQuant's `Generator<Context>` plug-in already ships a NumPy backend (`python_einsum.hpp`,
`NumPyEinsumGenerator`). Emitting it for this residual is a ~20-line additive block in the derivation tool
(`sequant-fork/tests/manual/test_csv_ccsd_derivation.cpp`), mirroring the existing CTIR and TiledArray
blocks, exporting from a **copy of the same cache-free (`SPTC_NO_CSE=1`) forest**:

```cpp
auto np_forest = forest;                 // copy; the TiledArray export moves the original
NumPyEinsumGenerator np_gen;
NumPyEinsumGeneratorContext np_ctx;
export_group(ExpressionGroup<ExportExpr>{std::move(np_forest), fn_name}, np_gen, np_ctx);
// -> /tmp/claude-ta-generator-test/generated_R{1,2}.py
```

The result is a NumPy program (committed examples: `backends/numpy/generated_t{1,2}_residual.py`) whose
`np.einsum` **subscripts are correct** — the einsum-spec builder assigns distinct per-call letters to
distinct indices (e.g. `np.einsum('iabc,dc->iabd', ...)`), so the *contraction structure* is faithful to the
same sequence the TiledArray backend runs. **This is the framework-agnosticism proof: one forest, two
backends, same sequence.**

## What does NOT yet work: the stock generator's CSV/PNO gaps (the finding)

Running the emitted program on the real ethane leaves is **not** yet possible with the stock generator. Three
concrete gaps, all in the generator's *per-space* naming/shaping (not in our block, and not in the sequence):

1. **μ̃ and Κ share a tag byte.** Tensor names use `get_tag(space) = base_key[0]` — the first *UTF-8 byte*.
   `μ̃` (`0xCE 0xBC`) and `Κ` (`0xCE 0x9A`) both start `0xCE`, so both tag as the same byte (renders `�`).
   Leaf names like `g_i��` are still individually unique (index *positions* differ), but the tags are not a
   valid, distinct, ASCII-safe identifier scheme.
2. **`dim_a` conflates the two PNO families.** The CSV singles PNO (`ap1`, extent **647**) and doubles PNO
   (`ap2`, extent **375**) are the *same* base space `a`, so every PNO axis is shaped `dim_a`. A single
   `dim_a` value cannot be both 647 and 375, so the pre-zeroed intermediates (`np.zeros((.., dim_a, dim_a))`)
   are dimensionally ambiguous.
3. **Lossy ToT/proto leaf names.** The CSV coefficient tensors are tensors-of-tensors (outer + inner-`a`
   proto structure). They emit with too few tags — e.g. the 3-index `C(i,μ̃;a)` and 4-index `C(i,i,μ̃;a)`
   both surface as 2-tag `C_�a`/`C_a�` — so a generated leaf name cannot be mapped back to a specific `.tns`
   file. This is the same limitation the repo already documents at the IR level: **flat einsum cannot fully
   describe MPQC's CSV/PNO tensor-of-tensor computation** (`docs/CONTRACTION_IR.md`,
   `docs/MPQC_EVALUATION.md`).

Gaps 1–2 are ordinary generator polish (assign **per-index**, not per-space, tags/extents). Gap 3 is the
substantive one: a flat-einsum framework has no native ragged-inner-index (ToT) construct, so a *faithful*
whole-residual run needs either (a) the generator to lower the ToT to explicit per-pair einsum families, or
(b) a target framework with blocked/ragged tensors. Either is a real follow-on, deliberately **not** done
here — doing it inside the harness is how the effort previously drifted toward reimplementing a framework.

## How to run it (once the generator emits per-index metadata)

`tools/numpy_runner.py` is the harness: it converts the `.tns` COO leaves to dense `.npy`, binds the index
extents, executes the generated program, and computes the same `nnz/sum/sumsq/max_abs` checksum as the TA
path (`src/ta_dumper.h`). It runs end-to-end for the flat leaves (`f`, `g`, `s`, `t` with an axis-permute)
and prints a precise diagnostic for the unresolved CSV `C`/PNO leaves, pointing back here — so it becomes a
complete cross-framework check the moment gaps 1–3 are addressed. See its header for the leaf map and the
dense-vs-ragged caveat.

## Bottom line

- **Delivered:** the multi-backend export is real and one forest drives it; the NumPy *sequence* is faithful;
  a converter + runner scaffold and committed example programs.
- **Honest gap:** the stock NumPy generator needs per-index tags/extents and ToT lowering to *run* the CSV
  residual — a scoped generator enhancement, cleanly identified, not silently worked around.
- **For a different framework** (PyTorch, Julia): the same additive-block recipe applies
  (`PyTorchEinsumGenerator`, `julia_*` generators already exist); the CSV/PNO caveat above is the general
  requirement for any *flat* backend.
