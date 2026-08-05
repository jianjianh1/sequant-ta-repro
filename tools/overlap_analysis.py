#!/usr/bin/env python3
"""Step 2 -- realized-overlap analysis for MPQC's CCSD residual.

Question: does MPQC overlap the evaluation of its residual's contractions?

Method: compare the *asynchronous* whole-residual wall (TiledArray's normal
scheduling, fenced only at the ends -- ``MPQC_NATIVE_RESIDUAL_CSV``) against the
strictly serialized sum of that same residual's own per-product completion walls
(from the level-4 graph trace of the same process, same amplitudes, same
``cache=none``).

    G = T_async - Sigma_products

If MPQC overlapped products, the async residual would finish in *less* than the
serialized sum and G would be negative.  Every measurement artifact pushes the
same way -- Sigma_products omits accumulation, symmetrization and inter-product
gaps, which deflates the serial side -- so the comparison is handicapped in
favour of finding overlap.  A positive G is therefore unambiguous.

Only np == 1 supports a quantitative claim: the graph trace is rank-0 local and
not gop.max reduced, so at np > 1 Sigma_products understates the max-rank
critical path.  np > 1 rows are printed but labelled DIRECTIONAL.

Usage:  python3 overlap_analysis.py [--all-np] [--csv out.csv]
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys

import equation_parallelism_data as D

# Accumulation cost, from a genuine cache_imeds=false run (mpqc-benchmark
# work/ethane-cachefree-perf.log).  NOTE: measured with emit_diagnostics=true,
# which fences after *every* += (cck.ipp:1833), so these are UPPER bounds on the
# real accumulation cost.  Basis is cc-pVDZ-F12/aug-cc-pVDZ-RI, whereas the
# campaign is cc-pVTZ/cc-pVTZ-RI -- so this bounds, and does not measure, the
# campaign's cells.
ACCUM_LOG = "/users/jianjian/mpqc-benchmark/work/ethane-cachefree-perf.log"


def load_accumulation_bound(path: str = ACCUM_LOG) -> list[dict]:
    """Parse ``Eval | SumInplace`` / ``WholeResidualWallTime`` pairs."""
    pat = re.compile(
        r"Eval \| (SumInplace|WholeResidualWallTime) \| (\d+)ns \| R=(\d+) \| n_terms=(\d+)"
    )
    seen: list[dict] = []
    pending: dict[str, int] = {}
    with open(path) as fh:
        for line in fh:
            m = pat.search(line)
            if not m:
                continue
            kind, ns, R, n_terms = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
            if kind == "SumInplace":
                pending[R] = ns
            elif R in pending:
                seen.append(
                    {
                        "residual": f"R{R}",
                        "n_terms": n_terms,
                        "accum_ns": pending.pop(R),
                        "whole_ns": ns,
                    }
                )
    for s in seen:
        s["accum_pct"] = 100.0 * s["accum_ns"] / s["whole_ns"]
    return seen


def trial_drift(trials: list[tuple[int, float]]) -> float | None:
    """last/first trial ratio -- bounds the 'later passes run slower' confound."""
    if len(trials) < 2:
        return None
    return trials[-1][1] / trials[0][1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-np", action="store_true", help="include directional np>1 rows")
    ap.add_argument("--csv", help="write the per-cell table here")
    args = ap.parse_args()

    traces = D.load_all_graphs()
    summary = D.load_summary()
    trials = D.load_trials()
    prov = D.load_provenance()

    problems: list[str] = []
    for tr in traces.values():
        problems += tr.check()
    if problems:
        print("STRUCTURAL CHECK FAILED -- refusing to report:", file=sys.stderr)
        for p in problems:
            print("  " + p, file=sys.stderr)
        return 1

    rows = []
    for (mol, np_, residual), tr in sorted(traces.items()):
        key = (mol, np_, residual)
        if key not in summary:
            continue
        D.assert_same_cell(key, key)  # same-np lint: numerator == denominator cell
        s = summary[key]
        sigma = tr.products_s
        t_async = s["wall_median_s"]
        rows.append(
            {
                "molecule": mol,
                "np": np_,
                "residual": residual,
                "n_terms": len(tr.terms),
                "n_products": tr.n_products,
                "sigma_products_s": sigma,
                "t_async_s": t_async,
                "ratio": t_async / sigma,
                "G_s": t_async - sigma,
                "G_pct": 100.0 * (t_async - sigma) / sigma,
                "leaves_s": tr.leaves_s,
                "permutes_s": tr.permutes_s,
                "wall_cv": s["wall_cv"],
                "trials": s["trials"],
                "drift": trial_drift(trials.get(key, [])),
                "quantitative": np_ == 1,
            }
        )

    hdr = (
        f"{'mol':7}{'R':4}{'np':>4}  {'terms':>5}{'prods':>6}"
        f"{'Sigma_prod_s':>14}{'T_async_s':>11}{'ratio':>8}{'G_s':>9}{'G%':>7}"
        f"{'CV%':>7}{'drift':>7}"
    )

    print("=" * 92)
    print("Realized overlap: asynchronous whole-residual wall vs. serialized product sum")
    print("=" * 92)
    mx = prov.get("matrix", {})
    hosts = prov.get("hostfile", {}).get("hosts", [])
    print(f"build_id      : {prov.get('build_id', '?')}")
    print(f"sif sha256    : {prov.get('sif', {}).get('sha256', '?')[:16]}...")
    print(f"backend       : {mx.get('backend', '?')}   "
          f"threads/rank: {mx.get('threads_per_rank', '?')}   "
          f"ranks/node: {mx.get('ranks_per_node', '?')}   "
          f"trials: {mx.get('trials', '?')}")
    print(f"hosts         : {len(hosts)} x {hosts[0] if hosts else '?'}"
          f"{'..' + hosts[-1] if len(hosts) > 1 else ''}")
    print(f"equation set  : {D.EQUATION_SET_SHA256[:16]}...  "
          f"({D.PRODUCTS_PER_GRAPH} products, "
          f"{D.TERMS_PER_RESIDUAL['R1']}+{D.TERMS_PER_RESIDUAL['R2']} terms)")
    print("cache_mode    : none (cache_imeds=false), state=post_solve_converged")
    print()
    print("QUANTITATIVE (np=1 -- trace is rank-0 local, so only np=1 is admissible)")
    print(hdr)
    print("-" * 92)

    def emit(r):
        d = f"{r['drift']:7.3f}" if r["drift"] is not None else f"{'-':>7}"
        print(
            f"{r['molecule']:7}{r['residual']:4}{r['np']:>4}  "
            f"{r['n_terms']:>5}{r['n_products']:>6}"
            f"{r['sigma_products_s']:>14.3f}{r['t_async_s']:>11.3f}"
            f"{r['ratio']:>8.3f}{r['G_s']:>9.3f}{r['G_pct']:>7.1f}"
            f"{100 * r['wall_cv']:>7.2f}{d}"
        )

    q = [r for r in rows if r["quantitative"]]
    for r in q:
        emit(r)

    ratios = [r["ratio"] for r in q]
    print("-" * 92)
    print(
        f"np=1 ratio: min {min(ratios):.3f}  median {statistics.median(ratios):.3f}  "
        f"max {max(ratios):.3f}   (n={len(ratios)} cells)"
    )
    neg = [r for r in q if r["G_s"] <= 0]
    print()
    print("VERDICT")
    if neg:
        print(f"  {len(neg)} of {len(q)} np=1 cells show G <= 0 (overlap detected):")
        for r in neg:
            print(f"    {r['molecule']} {r['residual']}: G = {r['G_s']:+.3f} s")
    else:
        print(f"  G > 0 in ALL {len(q)} np=1 cells.  The asynchronous residual costs")
        print(f"  {min(r['G_pct'] for r in q):.1f}-{max(r['G_pct'] for r in q):.1f}% MORE "
              "than the strictly serialized sum of its own")
        print("  products.  MPQC realizes ZERO contraction-level overlap, and pays that")
        print("  much again in residual-level overhead.  The comparison is handicapped in")
        print("  favour of finding overlap, so this direction is unambiguous.")

    # Where the residual-level overhead goes.
    print()
    print("RESIDUAL-LEVEL OVERHEAD BREAKDOWN (np=1, as % of Sigma_products)")
    print(f"{'mol':7}{'R':4}{'G%':>8}{'leaves%':>9}{'permutes%':>11}{'unexplained%':>14}")
    print("-" * 53)
    for r in q:
        lp = 100.0 * r["leaves_s"] / r["sigma_products_s"]
        pp = 100.0 * r["permutes_s"] / r["sigma_products_s"]
        print(
            f"{r['molecule']:7}{r['residual']:4}{r['G_pct']:>8.1f}{lp:>9.3f}"
            f"{pp:>11.2f}{r['G_pct'] - lp - pp:>14.1f}"
        )
    print()
    print("  Leaf loads are negligible; the closing per-term permutations are small but")
    print("  real.  The unexplained remainder is accumulation, R2 symmetrization, the")
    print("  trailing fences, and deferred TiledArray cleanup drained between products.")

    # Accumulation bound.
    print()
    print("ACCUMULATION COST (upper bound; cc-pVDZ-F12, emit_diagnostics=true so a")
    print("gop.fence() follows every += -- cck.ipp:1833)")
    print(f"{'residual':10}{'terms':>6}{'accum_ms':>10}{'whole_s':>10}{'accum%':>9}")
    print("-" * 45)
    for a in load_accumulation_bound():
        print(
            f"{a['residual']:10}{a['n_terms']:>6}{a['accum_ns'] / 1e6:>10.2f}"
            f"{a['whole_ns'] / 1e9:>10.3f}{a['accum_pct']:>9.3f}"
        )
    print()
    print("  The 55-term += chain is well under 1% of the residual even when fenced")
    print("  after every term.  It is NOT a performance target.  Its only significance")
    print("  for this study is structural: because += makes the LHS an input of the RHS")
    print("  expression (upstream TiledArray expressions/tsr_expr.h:150), the chain")
    print("  cannot consume terms out of order, which is what an out-of-order or")
    print("  concurrent equation schedule would need.")

    if args.all_np:
        print()
        print("DIRECTIONAL ONLY (np>1: trace is rank-0 local, NOT gop.max reduced, so")
        print("Sigma_products understates the max-rank critical path -- do not quote)")
        print(hdr)
        print("-" * 92)
        for r in rows:
            if not r["quantitative"]:
                emit(r)

    if args.csv:
        import csv as _csv

        with open(args.csv, "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv} ({len(rows)} rows)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
