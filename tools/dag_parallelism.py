#!/usr/bin/env python3
"""Step 3 -- available-parallelism ceiling for MPQC's CCSD residual.

Reconstructs the true dependence structure of the 81 residual terms / 727 binary
products, weights it with the measured per-product completion walls, and reports
how much concurrency is *available* at two granularities:

  equation level      -- schedule whole terms onto N workers (LPT bin-packing).
                         Terms are mutually independent, so this is pure packing.
  contraction level   -- schedule individual products onto N workers, respecting
                         each term's internal binary tree (greedy list schedule).

Structure comes from the SHA-pinned execution plans
(mpqc-benchmark/catalogs/native-no-cache-v1/execution-plans/*.json); walls come
from the level-4 graph traces.  Both are the same post-order op sequence, so they
zip position-by-position -- asserted, not assumed.

Term independence is a *consequence of the measurement contract*, not an
assumption: with ``cache_imeds=false`` no intermediate is reused across terms, so
every term's leaves are input tensors and the only cross-term edge is the
accumulation into R (a reduction).  The script asserts there are no I(...) leaves.

CAVEAT PRINTED WITH EVERY CEILING: the weights are 8-thread walls.  Using them as
if each of N concurrent workers also had 8 threads over-counts the machine by N x.
These are ceilings at FIXED PER-WORKER thread budget, not at fixed total budget.
Correcting that needs a thread scan (wall at 1/2/4 threads), which this study's
"existing data only" scope forgoes.  See docs/EQUATION_PARALLELISM.md.

Usage:  python3 dag_parallelism.py [--molecule C4H10] [--csv out.csv]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import equation_parallelism_data as D

WORKER_COUNTS = (2, 4, 8, 16)
_OPERANDS = re.compile(r"^(.*?) \* (.*?) -> (\S+)$")


# ---------------------------------------------------------------- reconstruction


class Node:
    """A node in one term's binary evaluation tree."""

    __slots__ = ("kind", "name", "cost", "left", "right", "idx")

    def __init__(self, kind, name, cost=0.0, left=None, right=None, idx=-1):
        self.kind = kind  # "leaf" | "product"
        self.name = name
        self.cost = cost  # seconds; leaves are ~0
        self.left = left
        self.right = right
        self.idx = idx  # position among this term's products

    def critical_path(self) -> float:
        """Heaviest weighted root-to-leaf path (the term's span)."""
        if self.kind == "leaf":
            return 0.0
        return self.cost + max(self.left.critical_path(), self.right.critical_path())

    def depth(self) -> int:
        if self.kind == "leaf":
            return 0
        return 1 + max(self.left.depth(), self.right.depth())

    def products(self):
        if self.kind == "leaf":
            return
        yield from self.left.products()
        yield from self.right.products()
        yield self


class TermTree:
    def __init__(self, term_id: str, root: Node, leaf_names: list[str]):
        self.term_id = term_id
        self.root = root
        self.leaf_names = leaf_names
        self._products = list(root.products())

    @property
    def n_products(self) -> int:
        return len(self._products)

    @property
    def work(self) -> float:
        return sum(p.cost for p in self._products)

    @property
    def span(self) -> float:
        return self.root.critical_path()

    @property
    def depth(self) -> int:
        return self.root.depth()

    def product_list(self):
        return self._products


def reconstruct(plan_ops: list[dict], trace_term) -> TermTree:
    """Stack-simulate one term's post-order op list into a binary tree.

    Tensor/Constant push a leaf; Product pops two and pushes; Permute is a no-op
    on the single remaining element.  The popped pair's names must match the
    Product's printed operand annotation -- asserted.
    """
    stack: list[Node] = []
    leaves: list[str] = []
    prod_i = 0
    walls = trace_term.product_ns

    for op in plan_ops:
        kind, expr = op["op"], op["expression"]
        if kind in ("Tensor", "Constant"):
            stack.append(Node("leaf", expr))
            leaves.append(expr)
        elif kind == "Product":
            m = _OPERANDS.match(expr)
            if not m:
                raise AssertionError(f"unparseable Product expression: {expr!r}")
            lhs, rhs, res = m.group(1), m.group(2), m.group(3)
            if len(stack) < 2:
                raise AssertionError(f"stack underflow at {expr!r}")
            right = stack.pop()
            left = stack.pop()
            # The stack order must agree with the printed annotation.
            if left.name != lhs or right.name != rhs:
                raise AssertionError(
                    f"operand mismatch: stack ({left.name!r}, {right.name!r}) "
                    f"vs annotation ({lhs!r}, {rhs!r})"
                )
            if prod_i >= len(walls):
                raise AssertionError("more plan products than traced walls")
            stack.append(
                Node("product", res, walls[prod_i] / 1e9, left, right, prod_i)
            )
            prod_i += 1
        elif kind == "Permute":
            if len(stack) != 1:
                raise AssertionError(
                    f"Permute with {len(stack)} items on the stack, expected 1"
                )
        else:
            raise AssertionError(f"unknown op kind {kind!r}")

    if len(stack) != 1:
        raise AssertionError(f"term ended with {len(stack)} roots, expected 1")
    if prod_i != len(walls):
        raise AssertionError(
            f"traced {len(walls)} product walls but plan has {prod_i} products"
        )
    return TermTree("", stack[0], leaves)


def build(molecule: str, np_: int = 1):
    """Reconstruct all 81 term trees for one molecule, weighted by measured walls."""
    path = os.path.join(D.CATALOG, "execution-plans", f"{molecule}.json")
    with open(path) as fh:
        plan = json.load(fh)

    traces = {
        r: t
        for r, t in (
            (res, D.load_all_graphs().get((molecule, np_, res)))
            for res in ("R1", "R2")
        )
        if t is not None
    }
    if len(traces) != 2:
        raise SystemExit(f"{molecule} np{np_}: missing a residual trace")

    # Plan terms are R1.T001..R1.T026 then R2.T001..R2.T055, in graph order.
    by_residual: dict[str, list[TermTree]] = {"R1": [], "R2": []}
    plan_terms = plan["terms"]
    cursor = {"R1": 0, "R2": 0}
    for pt in plan_terms:
        residual = pt["term_id"].split(".")[0]
        i = cursor[residual]
        trace_term = traces[residual].terms[i]
        tree = reconstruct(pt["operations"], trace_term)
        tree.term_id = pt["term_id"]
        by_residual[residual].append(tree)
        cursor[residual] += 1

    for residual, trees in by_residual.items():
        want = D.TERMS_PER_RESIDUAL[residual]
        if len(trees) != want:
            raise AssertionError(f"{residual}: {len(trees)} terms, expected {want}")

    return plan, by_residual


# ---------------------------------------------------------------- scheduling


def lpt_makespan(costs: list[float], n_workers: int) -> float:
    """Longest-processing-time-first bin packing.  Terms are independent."""
    loads = [0.0] * n_workers
    for c in sorted(costs, reverse=True):
        i = min(range(n_workers), key=lambda k: loads[k])
        loads[i] += c
    return max(loads)


def list_schedule(trees: list[TermTree], n_workers: int) -> float:
    """Greedy list schedule over the product DAG with dependencies.

    Each product becomes ready when both operands are done.  Workers take the
    heaviest ready product.  This is Graham's list scheduling; communication and
    scheduling overhead are modelled as zero, so the result is an upper bound.
    """
    # Flatten: every product gets a unique id, with edges from operand products.
    ready_at: dict[int, float] = {}
    preds: dict[int, list[int]] = {}
    cost: dict[int, float] = {}
    nid = 0
    for t in trees:
        local: dict[int, int] = {}
        for p in t.product_list():
            local[id(p)] = nid
            deps = [
                local[id(c)]
                for c in (p.left, p.right)
                if c is not None and c.kind == "product"
            ]
            preds[nid] = deps
            cost[nid] = p.cost
            nid += 1

    done: dict[int, float] = {}
    remaining = set(preds)
    worker_free = [0.0] * n_workers
    # Event-driven greedy: repeatedly pick the earliest-free worker and give it
    # the heaviest product whose predecessors have all finished by then.
    while remaining:
        w = min(range(n_workers), key=lambda k: worker_free[k])
        t_now = worker_free[w]
        eligible = [
            n for n in remaining if all(p in done and done[p] <= t_now for p in preds[n])
        ]
        if not eligible:
            # No product ready yet: advance this worker to the next completion.
            future = [
                max(done[p] for p in preds[n]) if preds[n] else 0.0
                for n in remaining
                if all(p in done for p in preds[n])
            ]
            if not future:
                future = [min(v for v in done.values())] if done else [0.0]
            worker_free[w] = max(min(future), t_now + 1e-12)
            continue
        pick = max(eligible, key=lambda n: cost[n])
        start = t_now
        finish = start + cost[pick]
        done[pick] = finish
        worker_free[w] = finish
        remaining.discard(pick)
    return max(worker_free)


# ---------------------------------------------------------------- reporting


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--molecule", action="append", help="default: all with traces")
    ap.add_argument("--csv", help="write the per-cell ceiling table here")
    args = ap.parse_args()

    molecules = args.molecule or ["C2H6", "C3H8", "C4H10", "C5H12"]
    rows = []

    print("=" * 100)
    print("Available parallelism in MPQC's CCSD residual (np=1, 8 threads/rank)")
    print("=" * 100)
    print("Structure: SHA-pinned execution plans.  Weights: measured per-product walls.")
    print()

    for mol in molecules:
        try:
            plan, by_residual = build(mol)
        except (FileNotFoundError, SystemExit) as e:
            print(f"{mol}: skipped ({e})")
            continue

        # --- structural gates -------------------------------------------------
        n_terms = sum(len(v) for v in by_residual.values())
        n_prod = sum(t.n_products for v in by_residual.values() for t in v)
        if n_terms != 81 or n_prod != D.PRODUCTS_PER_GRAPH:
            print(f"{mol}: STRUCTURAL FAILURE {n_terms} terms / {n_prod} products",
                  file=sys.stderr)
            return 1
        leaf_kinds: dict[str, int] = {}
        for v in by_residual.values():
            for t in v:
                for ln in t.leaf_names:
                    head = ln.split("(")[0] if "(" in ln else "Constant"
                    leaf_kinds[head] = leaf_kinds.get(head, 0) + 1
        if "I" in leaf_kinds:
            print(f"{mol}: FAILURE -- {leaf_kinds['I']} intermediate leaves; terms are "
                  "NOT independent", file=sys.stderr)
            return 1

        print(f"--- {mol}  (execution_plan_sha256 "
              f"{plan['execution_plan_sha256'][:16]}...) ---")
        print(f"  reconstruction: {n_terms}/81 terms, {n_prod}/{D.PRODUCTS_PER_GRAPH} "
              "products, every operand pair matched")
        print(f"  leaf census   : "
              + ", ".join(f"{k}:{v}" for k, v in sorted(leaf_kinds.items()))
              + "  (no I(...) leaves -> terms are independent)")
        print()

        for residual in ("R1", "R2"):
            trees = by_residual[residual]
            work = sum(t.work for t in trees)
            span = max(t.span for t in trees)  # forest: span = heaviest term's path
            term_costs = [t.work for t in trees]
            biggest = max(trees, key=lambda t: t.work)
            biggest_prod = max(
                (p for t in trees for p in t.product_list()), key=lambda p: p.cost
            )

            print(f"  {residual}: work {work:.3f} s over {len(trees)} terms / "
                  f"{sum(t.n_products for t in trees)} products")
            print(f"      span (heaviest term's critical path) {span:.3f} s   "
                  f"work/span {work / span:.2f}")
            print(f"      largest term    {biggest.term_id}: {biggest.work:.3f} s = "
                  f"{100 * biggest.work / work:.1f}% of work "
                  f"({biggest.n_products} products, depth {biggest.depth})")
            print(f"      largest product : {biggest_prod.cost:.3f} s = "
                  f"{100 * biggest_prod.cost / work:.1f}% of work")
            print(f"      term depth      : min {min(t.depth for t in trees)}, "
                  f"max {max(t.depth for t in trees)}, "
                  f"mean {sum(t.depth for t in trees) / len(trees):.2f}")

            print(f"      {'N':>3}  {'equation-level':>16}  {'contraction-level':>18}")
            row = {
                "molecule": mol,
                "residual": residual,
                "work_s": work,
                "span_s": span,
                "work_over_span": work / span,
                "n_terms": len(trees),
                "n_products": sum(t.n_products for t in trees),
                "largest_term": biggest.term_id,
                "largest_term_pct": 100 * biggest.work / work,
                "largest_product_pct": 100 * biggest_prod.cost / work,
            }
            for n in WORKER_COUNTS:
                eq = work / lpt_makespan(term_costs, n)
                ct = work / list_schedule(trees, n)
                print(f"      {n:>3}  {eq:>15.2f}x  {ct:>17.2f}x")
                row[f"eq_speedup_N{n}"] = eq
                row[f"contr_speedup_N{n}"] = ct
            rows.append(row)
            print()

    print("=" * 100)
    print("CAVEAT -- read before quoting any number above")
    print("=" * 100)
    print("These are ceilings at FIXED PER-WORKER thread budget.  The weights are")
    print("8-thread walls, so treating N concurrent workers as each having 8 threads")
    print("over-counts the machine by N x.  On the 8-core Xeon D-1548 these runs used,")
    print("N concurrent workers would realistically get 8/N threads each.  Converting")
    print("to a fixed TOTAL budget needs f(t) = wall(t threads)/wall(8 threads), which")
    print("this study does not measure.  Treat these as upper bounds on AVAILABLE")
    print("parallelism, not as predicted speedups.")
    print()
    print("Also modelled as zero: communication, scheduling overhead, memory-bandwidth")
    print("and L3 contention between concurrent workers.  Single traced pass per cell,")
    print("so no variance.  np=1 only (the trace is rank-0 local, not gop.max reduced).")

    if args.csv and rows:
        import csv as _csv

        with open(args.csv, "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv} ({len(rows)} rows)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
