"""Shared loaders for the equation-parallelism characterization study.

Reads only artifacts already on disk -- the level-4 native residual graph traces
and the sweep CSVs under the campaign directory.  No machine time, no MPQC runs.

The graph trace is emitted by ``CCk::run_native_residual_benchmark``'s
``export_after_timing`` pass (mpqc4 ``cck.ipp``) at ``logger.eval.level = 4``.
Its structure, verified against C2H6..C5H12:

    MPQC_NATIVE_RESIDUAL_GRAPH_R_BEGIN,R=<n>
      Eval | Tensor  | <ns>ns | ...        <- leaf load
      Eval | Constant| <ns>ns | ...        <- scalar
      Eval | Product | <ns>ns | ... | <lhs> * <rhs> -> <result>
      Eval | Permute | <ns>ns | ...        <- ends a term (one per term)
      MPQC_NATIVE_RESIDUAL_GRAPH_OP,R=<n>,op=SumInplace,term=<k>
      ...
      MPQC_NATIVE_RESIDUAL_GRAPH_OP,R=<n>,op=Symmetrize   (R2 only)
    MPQC_NATIVE_RESIDUAL_GRAPH_R_END,R=<n>

Each ``Eval | <kind> | <ns>ns`` figure is a *completion* time, not a submission
time: ``detail::timed_eval_inplace`` (sequant-fork SeQuant/core/eval/eval.hpp)
wraps the TiledArray statement, and TA's expression assignment ends in
``dist_eval.wait()`` (upstream TiledArray expressions/expr.h:424), which blocks
until the statement's local output tiles are computed.  The rss()/log formatting
happens after the timer stops, so trace overhead is outside the interval.

IMPORTANT (np > 1): the trace is written by rank 0 only and is NOT gop.max
reduced, so per-product sums at np > 1 understate the max-rank critical path.
Only np == 1 supports a quantitative claim; loaders tag every record with
``rank0_local`` so callers cannot forget.
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass, field

CAMPAIGN = "/proj/perf-model-gpu-PG0/native-residual-c2-c6-node16-node31-20260804"
CATALOG = "/users/jianjian/mpqc-benchmark/catalogs/native-no-cache-v1"

# Term counts fixed by the pinned equation set.
TERMS_PER_RESIDUAL = {"R1": 26, "R2": 55}
PRODUCTS_PER_GRAPH = 727
EQUATION_SET_SHA256 = (
    "5ee7844d206b4da1c51d7f1264637961f043dccce2cb732e57609f65edc34410"
)

_EVAL = re.compile(r"^Eval \| (Tensor|Constant|Product|Permute) \| (\d+)ns")
_R_BEGIN = re.compile(r"^MPQC_NATIVE_RESIDUAL_GRAPH_R_BEGIN,R=(\d+)")
_R_END = re.compile(r"^MPQC_NATIVE_RESIDUAL_GRAPH_R_END,R=(\d+)")
_OP = re.compile(r"^MPQC_NATIVE_RESIDUAL_GRAPH_OP,R=(\d+),op=(\w+)")
# Product operand annotation, e.g. "g(i,j,K) * g(m,n,K) -> I(i,j,n,m)"
_OPERANDS = re.compile(r"\| ([^|]+) \* ([^|]+) -> (\S+)\s*$")


@dataclass
class Term:
    """One residual term: its product walls plus the closing permute."""

    ordinal: int  # 1-based within the residual
    product_ns: list[int] = field(default_factory=list)
    tensor_ns: list[int] = field(default_factory=list)
    constant_ns: list[int] = field(default_factory=list)
    permute_ns: int = 0
    operands: list[tuple[str, str, str]] = field(default_factory=list)

    @property
    def n_products(self) -> int:
        return len(self.product_ns)

    @property
    def product_s(self) -> float:
        """Sum of this term's product completion walls, seconds."""
        return sum(self.product_ns) / 1e9

    @property
    def total_s(self) -> float:
        """Products + leaf loads + constants + the closing permute, seconds."""
        return (
            sum(self.product_ns)
            + sum(self.tensor_ns)
            + sum(self.constant_ns)
            + self.permute_ns
        ) / 1e9


@dataclass
class ResidualTrace:
    molecule: str
    np: int
    residual: str  # "R1" | "R2"
    terms: list[Term] = field(default_factory=list)
    ops: list[str] = field(default_factory=list)  # SumInplace / Symmetrize markers
    rank0_local: bool = True  # trace is rank-0 local, never gop.max reduced

    @property
    def n_products(self) -> int:
        return sum(t.n_products for t in self.terms)

    @property
    def products_s(self) -> float:
        """Strictly serialized sum of every product's completion wall."""
        return sum(t.product_s for t in self.terms)

    @property
    def leaves_s(self) -> float:
        return sum((sum(t.tensor_ns) + sum(t.constant_ns)) / 1e9 for t in self.terms)

    @property
    def permutes_s(self) -> float:
        return sum(t.permute_ns / 1e9 for t in self.terms)

    @property
    def n_sum_inplace(self) -> int:
        return sum(1 for o in self.ops if o == "SumInplace")

    def check(self) -> list[str]:
        """Structural assertions.  Returns a list of problems (empty == clean)."""
        problems = []
        want_terms = TERMS_PER_RESIDUAL[self.residual]
        if len(self.terms) != want_terms:
            problems.append(
                f"{self.molecule} np{self.np} {self.residual}: "
                f"{len(self.terms)} terms, expected {want_terms}"
            )
        # Terms 2..n each get one SumInplace marker; term 1 initializes the target.
        if self.n_sum_inplace != want_terms - 1:
            problems.append(
                f"{self.molecule} np{self.np} {self.residual}: "
                f"{self.n_sum_inplace} SumInplace markers, expected {want_terms - 1}"
            )
        if self.residual == "R2" and "Symmetrize" not in self.ops:
            problems.append(
                f"{self.molecule} np{self.np} R2: missing Symmetrize marker"
            )
        for t in self.terms:
            if t.n_products == 0:
                problems.append(
                    f"{self.molecule} np{self.np} {self.residual} "
                    f"term {t.ordinal}: zero products"
                )
        return problems


def parse_graph(path: str, molecule: str, np_: int) -> dict[str, ResidualTrace]:
    """Segment one ``<mol>.np<n>.native-graph.txt`` into per-residual traces.

    Terms are delimited by ``Eval | Permute`` -- the final permutation into the
    residual's head layout, emitted exactly once per term.
    """
    out: dict[str, ResidualTrace] = {}
    cur: ResidualTrace | None = None
    term = Term(ordinal=1)

    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")

            m = _R_BEGIN.match(line)
            if m:
                cur = ResidualTrace(molecule, np_, f"R{m.group(1)}")
                term = Term(ordinal=1)
                continue

            if _R_END.match(line):
                if cur is not None:
                    out[cur.residual] = cur
                cur = None
                continue

            if cur is None:
                continue

            m = _OP.match(line)
            if m:
                cur.ops.append(m.group(2))
                continue

            m = _EVAL.match(line)
            if not m:
                continue
            kind, ns = m.group(1), int(m.group(2))

            if kind == "Product":
                term.product_ns.append(ns)
                om = _OPERANDS.search(line)
                if om:
                    term.operands.append(
                        (om.group(1).strip(), om.group(2).strip(), om.group(3).strip())
                    )
            elif kind == "Tensor":
                term.tensor_ns.append(ns)
            elif kind == "Constant":
                term.constant_ns.append(ns)
            elif kind == "Permute":
                # Permute closes the term.
                term.permute_ns = ns
                cur.terms.append(term)
                term = Term(ordinal=len(cur.terms) + 1)

    return out


def load_all_graphs() -> dict[tuple[str, int, str], ResidualTrace]:
    """Every graph trace in the campaign, keyed by (molecule, np, residual)."""
    traces: dict[tuple[str, int, str], ResidualTrace] = {}
    gdir = os.path.join(CAMPAIGN, "graphs")
    pat = re.compile(r"^(\w+)\.np(\d+)\.native-graph\.txt$")
    for name in sorted(os.listdir(gdir)):
        m = pat.match(name)
        if not m:
            continue
        mol, np_ = m.group(1), int(m.group(2))
        for residual, tr in parse_graph(os.path.join(gdir, name), mol, np_).items():
            traces[(mol, np_, residual)] = tr
    return traces


def load_summary() -> dict[tuple[str, int, str], dict]:
    """summary.csv: the asynchronous whole-residual wall (median over trials).

    This is the ``MPQC_NATIVE_RESIDUAL_CSV`` path: fenced only at the ends, so
    the residual runs with TiledArray's normal asynchronous scheduling.  It is
    gop.max reduced across ranks (cck.ipp does ``world.gop.max(wall_s)``).
    """
    rows: dict[tuple[str, int, str], dict] = {}
    with open(os.path.join(CAMPAIGN, "summary.csv")) as fh:
        for r in csv.DictReader(fh):
            if r["status"] != "OK":
                continue
            rows[(r["molecule"], int(r["np"]), r["residual"])] = {
                "wall_median_s": float(r["wall_median_s"]),
                "wall_mad_s": float(r["wall_mad_s"]),
                "wall_cv": float(r["wall_cv"]),
                "trials": int(r["trials"]),
                "peak_rss_kb": int(r["peak_rss_rank_max_kb"]),
                "graph_sha256": r["graph_sha256"],
            }
    return rows


def load_trials() -> dict[tuple[str, int, str], list[tuple[int, float]]]:
    """all_results.csv: per-trial async walls, for the trial-order drift control."""
    out: dict[tuple[str, int, str], list[tuple[int, float]]] = {}
    with open(os.path.join(CAMPAIGN, "all_results.csv")) as fh:
        for r in csv.DictReader(fh):
            if r["status"] != "OK":
                continue
            key = (r["molecule"], int(r["np"]), r["residual"])
            out.setdefault(key, []).append((int(r["trial"]), float(r["wall_s"])))
    for v in out.values():
        v.sort()
    return out


def load_provenance() -> dict:
    import json

    with open(os.path.join(CAMPAIGN, "provenance.json")) as fh:
        return json.load(fh)


def assert_same_cell(a: tuple, b: tuple) -> None:
    """Same-np lint.

    An earlier campaign compared np16 against np1 and had to be corrected.  Any
    ratio whose numerator and denominator differ in molecule, residual, or np is
    refused here rather than remembered.
    """
    if a != b:
        raise ValueError(
            f"refusing to form a ratio across cells: {a} vs {b}. "
            "Only same-(molecule, residual, np) comparisons are valid."
        )
