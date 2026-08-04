#!/usr/bin/env python3
"""Run the NumPy-einsum backend of the CSV-CCSD residual on real leaf data.

This is the second-framework runner (see docs/NUMPY_BACKEND.md): it executes the
SAME cache-free contraction sequence the TiledArray harness runs, but emitted for
NumPy from the same SeQuant forest (backends/numpy/generated_t{1,2}_residual.py).
It converts the .tns COO leaves to dense .npy, binds the index extents, execs the
generated program, and computes the same nnz/sum/sumsq/max_abs checksum as the TA
path (src/ta_dumper.h) so results are directly comparable across frameworks.

STATUS (honest): the flat leaves (f, g, s, t) map and load cleanly. The CSV
coefficient tensors (C_*) and the two PNO extents are NOT yet resolvable from the
stock generator's output -- it tags per index *space*, so (a) the singles/doubles
PNO families ap1(647)/ap2(375) collapse to one `dim_a`, and (b) the tensor-of-
tensor C leaves emit with too few tags to map back to a .tns file. Those are
generator enhancements (per-index tags/extents + ToT lowering), documented in
docs/NUMPY_BACKEND.md. Until then this runner loads what it can and reports the
exact blocker, so it becomes a complete cross-framework check once the generator
emits per-index metadata.

Usage:
    numpy_runner.py <residual: t1|t2> <leaf_dir> [--npy-dir DIR]
    e.g. numpy_runner.py t2 ../mpqc-benchmark/traces/checksum-run/sptc_coo_iter1
"""
import argparse
import os
import sys

import numpy as np

# Index-space extents for this ethane leaf set (from the .tns shape headers).
DIMS = {"dim_i": 9, "dim_μ̃": 114, "dim_Κ": 282, "dim_a": 647}  # dim_a: see caveat

# generated-leaf-name -> (.tns file, axis permutation to the generated index order).
# μ̃ and Κ both tag as the byte 0xCE (rendered below via \u-escapes to stay ASCII-
# safe in this source); positions still disambiguate the flat leaves.
# μ̃/Κ tag as the lone byte 0xCE (the generated .py is therefore not even valid
# UTF-8 -- gap #1 in docs/NUMPY_BACKEND.md). Read byte-preservingly (latin-1) so
# 0xCE -> U+00CE, and key the map on that.
CE = "\xce"
FLAT_LEAF_MAP = {
    "f_ii": ("f_i_1_i_2.tns", None),
    "f_i" + CE: ("f_i_1_m_1.tns", None),
    "f_" + CE + "i": ("f_i_1_m_1.tns", (1, 0)),          # f_μ̃_i = transpose(f_i_μ̃)
    "f_" + CE * 2: ("f_m_1_m_2.tns", None),
    "s_" + CE * 2: ("s_m_1_m_2.tns", None),
    "g_ii" + CE: ("g_i_1_i_2_Κ_1.tns", None),
    "g_i" + CE * 2: ("g_i_1_m_1_Κ_1.tns", None),
    "g_" + CE + "i" + CE: ("g_i_1_m_1_Κ_1.tns", (1, 0, 2)),  # g_μ̃_i_Κ
    "g_" + CE * 3: ("g_m_1_m_2_Κ_1.tns", None),
    "t_ai": ("t_i_1_a_1.tns", (1, 0)),                    # generated [a,i]
    "t_aaii": ("t_i_1_i_2_a_1_a_2.tns", (2, 3, 0, 1)),    # generated [a,a,i,i]
}


def load_tns_dense(path):
    """Load a .tns COO text file (# shape=... header + `idx.. value` rows) dense."""
    shape = None
    coords, vals = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                for tok in line.split():
                    if tok.startswith("shape="):
                        shape = tuple(int(x) for x in tok[6:].split(","))
                continue
            parts = line.split()
            coords.append(tuple(int(x) for x in parts[:-1]))
            vals.append(float(parts[-1]))
    if shape is None:
        raise ValueError(f"no shape= header in {path}")
    arr = np.zeros(shape, order="F")
    for c, v in zip(coords, vals):
        arr[c] = v
    return arr


def checksum(a):
    nz = a[a != 0.0]
    return dict(nnz=int(nz.size), sum=float(a.sum()),
                sumsq=float((a * a).sum()), max_abs=float(np.abs(a).max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("residual", choices=["t1", "t2"])
    ap.add_argument("leaf_dir")
    ap.add_argument("--npy-dir", default=None)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    prog = os.path.join(here, "..", "backends", "numpy",
                        f"generated_{args.residual}_residual.py")
    npy_dir = args.npy_dir or os.path.join(
        os.environ.get("TMPDIR", "/tmp"), "numpy_runner_leaves")
    os.makedirs(npy_dir, exist_ok=True)

    # Which leaves does the generated program load? Read byte-preservingly
    # (latin-1): the generated program contains lone 0xCE tag bytes and is not
    # valid UTF-8 (docs/NUMPY_BACKEND.md, gap #1).
    src = open(prog, encoding="latin-1").read()
    import re
    wanted = sorted(set(re.findall(r"np\.load\('([^']+)\.npy'\)", src)))

    unresolved = [w for w in wanted if w not in FLAT_LEAF_MAP]
    print(f"[numpy_runner] {args.residual}: {len(wanted)} leaves; "
          f"{len(wanted) - len(unresolved)} resolvable, {len(unresolved)} not.")

    # Convert the resolvable (flat) leaves.
    for name in wanted:
        if name not in FLAT_LEAF_MAP:
            continue
        fn, perm = FLAT_LEAF_MAP[name]
        arr = load_tns_dense(os.path.join(args.leaf_dir, fn))
        if perm is not None:
            arr = np.transpose(arr, perm).copy(order="F")
        np.save(os.path.join(npy_dir, name + ".npy"), arr)

    if unresolved:
        print("[numpy_runner] BLOCKED — the stock generator's CSV/PNO output is "
              "not yet runnable. Unresolved leaves (tensor-of-tensor C tensors "
              "emitted with per-space tags, and the ap1/ap2 PNO-extent conflation):")
        for u in unresolved:
            print(f"    {u!r}")
        print("[numpy_runner] See docs/NUMPY_BACKEND.md — this needs the generator "
              "to emit per-index (not per-space) tags/extents + ToT lowering. The "
              "flat leaves above were converted successfully, so the harness is "
              "ready the moment that lands.")
        return 2

    # (Reached only once the generator gaps are closed.) Run + checksum.
    ns = dict(DIMS)
    cwd = os.getcwd()
    os.chdir(npy_dir)
    try:
        exec(compile(src, prog, "exec"), ns)
        ns[f"whole_{args.residual}_residual"]()
        result = np.load(f"I_{'ia' if args.residual == 't1' else 'iiaa'}.npy")
    finally:
        os.chdir(cwd)
    cs = checksum(result)
    print(f"[numpy_runner] {args.residual} NumPy checksum: "
          f"nnz={cs['nnz']} sum={cs['sum']:.10g} sumsq={cs['sumsq']:.10g} "
          f"max_abs={cs['max_abs']:.10g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
