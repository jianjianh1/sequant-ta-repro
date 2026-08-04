#!/usr/bin/env python3
"""Post-process SeQuant's raw TiledArrayGenerator output into the committed
src/generated_t{1,2}_residual.cpp form.

The raw generator (tests/manual/test_csv_ccsd_derivation.cpp) hardcodes the
*arena* tensor-of-tensor type and emits no provenance header. The committed
files instead use the `ArrayToT` alias (which switches arena<->owning on the
SPTC_OWNING_TOT macro, so one source builds both backends) and carry a header.

This script makes that transform reproducible, so regenerating the cache-free
(SPTC_NO_CSE=1) sequence is a scripted step, not a hand-edit (README.md
"Regenerating"; the historical re-sync footgun). It does NOT touch the
leaf-parameter -> TATensors mapping in src/ta_sequant_native_residual.h -- the
parameter *order* changes on regeneration and that wrapper must be re-synced by
hand against the printed leaf manifest (see this script's --print-order output).

Usage:
    postprocess_generated.py <raw_Rr.cpp> <r> <date> [--print-order] > out.cpp
"""
import re
import sys

# The one ToT type string the raw generator emits (see ta_tensors.h: ArrayToT).
ARENA_TOT = "TA::DistArray<TA::Tensor<TA::ArenaTensor<double>>, TA::SparsePolicy>"

HEADER = """\
// AUTO-GENERATED ({date}) by SeQuant's native TiledArrayGenerator
// (SeQuant/core/export/tiledarray_generator.hpp) from the full
// cck.ipp-matching closed-shell CSV-CCSD T{r} residual derivation
// (tests/manual/test_csv_ccsd_derivation.cpp), emitted CACHE-FREE with
// SPTC_NO_CSE=1: the forest is exported un-deduped, per-summand, with NO
// cross-term common-subexpression elimination -- i.e. no reuse of shared
// intermediates across terms. This is the "without cache/reuse" contraction
// sequence the benchmark drives (README.md; docs/HARNESS_VS_EXPERIMENTS.md).
// The arena ToT type the generator hardcodes is rewritten to the ArrayToT
// alias (ta_tensors.h) so this one source builds both the owning and arena
// backends. Regenerate + re-run tools/postprocess_generated.py (do not
// hand-edit); re-sync src/ta_sequant_native_residual.h if the parameter order
// changed (see --print-order).
#include <tiledarray.h>
#include <TiledArray/expressions/einsum.h>
#include <cmath>
#include "ta_tensors.h"
"""


def main():
    raw_path, r, date = sys.argv[1], sys.argv[2], sys.argv[3]
    print_order = "--print-order" in sys.argv[4:]
    with open(raw_path, encoding="utf-8") as fh:
        code = fh.read()

    # Strip the raw include preamble (up to and including <cmath>); we re-emit
    # our own header + includes (which add ta_tensors.h).
    m = re.search(r"#include <cmath>\n", code)
    body = code[m.end():] if m else code

    if print_order:
        sig = re.search(r"whole_t\d+_residual\((.*?)\)\s*\{", body, re.S)
        params = sig.group(1) if sig else ""
        names = re.findall(r"&\s*([^,)]+?)\s*(?:,|$)", params)
        for i, name in enumerate(names, 1):
            print(f"  {i:2d}. {name.strip()}", file=sys.stderr)

    body = body.replace(ARENA_TOT, "ArrayToT")
    sys.stdout.write(HEADER.format(date=date, r=r))
    sys.stdout.write(body)


if __name__ == "__main__":
    main()
