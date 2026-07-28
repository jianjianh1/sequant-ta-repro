#!/usr/bin/env python3
"""SSA-split the generated T2 residual into a t-independent precompute
(persistent cache) + a t-dependent residual (MPQC's volatile/persistent
split). Reuses the einsum statements verbatim (only SSA-renames vars and
partitions), so correctness is checksum-verifiable."""
import re, os

SRC = "/users/jianjian/sequant-ta-repro/src/generated_t2_residual.cpp"
OUT = "/users/jianjian/sequant-ta-repro/gen_split/t2_split.cpp"
FLAT = "TA::TSpArrayD"
TOT = "ArrayToT"

src = open(SRC, encoding="utf-8").read()
sig = src[:src.index("{")]
body = src[src.index("{")+1: src.rindex("return")]
ret_var = re.search(r'return\s+([A-Za-z_][^\s;("]*)', src[src.rindex("return"):]).group(1)

params = []
for m in re.finditer(r'const\s+(TA::TSpArrayD|ArrayToT)\s*&\s*([A-Za-z_][^\s,)]*)', sig):
    params.append((m.group(2), m.group(1)))
leaf_names = {p[0] for p in params}
t_leaves = set(re.findall(r"\bt_[A-Za-z0-9_]+", sig))

localtype = {}
for m in re.finditer(r'^\s*(TA::TSpArrayD|ArrayToT)\s+([A-Za-z_][^\s;("]*)\s*;', body, re.M):
    localtype[m.group(2)] = "ArrayToT" in m.group(1)

lines = [l.strip() for l in body.splitlines() if l.strip()]
stmt_re = re.compile(r'([A-Za-z_][^\s("]*)\("([^"]*)"\)\s*(\+?=)\s*(.*)$', re.UNICODE)
ref_re = re.compile(r'([A-Za-z_][^\s("]*)\("', re.UNICODE)
rel_re = re.compile(r'^([A-Za-z_][^\s("]*)\s*=\s*(?:ArrayToT|TA::TSpArrayD)\(\)\s*;')

cur, ver = {}, 0
ssa_td, ssa_tot = {}, {}
stmts = []
for l in lines:
    if l.startswith("//"):
        continue
    if "// release" in l or rel_re.match(l):
        v = l.split("=")[0].strip().split("(")[0].strip(); cur.pop(v, None); continue
    m = stmt_re.match(l)
    if not m:
        continue
    lhs, annot, op, rhs = m.group(1), m.group(2), m.group(3), m.group(4)
    refs = [v for v in ref_re.findall(rhs) if v != "einsum"]
    refs_ssa, contrib_td = [], False
    for r in refs:
        if r in leaf_names:
            if r in t_leaves: contrib_td = True
            refs_ssa.append((r, None))
        elif r in cur:
            sid = cur[r]
            if ssa_td.get(sid, False): contrib_td = True
            refs_ssa.append((r, sid))
        else:
            refs_ssa.append((r, None))
    if op == "=":
        ver += 1; sid = ver; cur[lhs] = sid
        ssa_td[sid] = contrib_td; ssa_tot[sid] = localtype.get(lhs, True)
    else:
        sid = cur.get(lhs)
        if sid is None:
            ver += 1; sid = ver; cur[lhs] = sid; ssa_tot[sid] = localtype.get(lhs, True)
        ssa_td[sid] = ssa_td.get(sid, False) or contrib_td
    stmts.append({"ssa": sid, "op": op, "annot": annot, "rhs": rhs, "refs": refs_ssa})

for s in stmts:
    s["td"] = ssa_td[s["ssa"]]
ret_ssa = cur[ret_var]
name = lambda sid: f"v{sid}"
ctype = lambda sid: TOT if ssa_tot.get(sid, True) else FLAT

# boundary: t-indep ssa referenced by any t-dep statement
boundary = set()
for s in stmts:
    if s["td"]:
        for _, sid in s["refs"]:
            if sid is not None and not ssa_td[sid]:
                boundary.add(sid)

def rewrite_rhs(s):
    refs = list(s["refs"]); idx = [0]
    def repl(m):
        v = m.group(1)
        if v == "einsum":
            return m.group(0)
        _, sid = refs[idx[0]]; idx[0] += 1
        return (name(sid) if sid is not None else v) + '("'
    return re.sub(r'([A-Za-z_][^\s("]*)\("', repl, s["rhs"], flags=re.UNICODE)

prec, resid = [], []
decl_p, decl_r = set(), set()
for s in stmts:
    sid = s["ssa"]; lhs = name(sid); rhs2 = rewrite_rhs(s)
    tgt, decl = (resid, decl_r) if s["td"] else (prec, decl_p)
    if s["op"] == "=" and sid not in decl:
        tgt.append(f"  {ctype(sid)} {lhs};"); decl.add(sid)
    tgt.append(f'  {lhs}("{s["annot"]}") {s["op"]} {rhs2}')

cache_fields = "".join(f"  {ctype(s)} {name(s)};\n" for s in sorted(boundary))
# ToT boundary arrays must be arena-compacted before caching: their inner
# ArenaTensor cells are non-owning views into a transient arena slab that is
# freed when precompute() returns (and freed cross-thread by MADNESS), so the
# cached copy would dangle. arena_compact makes each tile a single owned page
# (exactly what MPQC does for persistent CSV coeffs, csv.ipp compact_csv_coeffs).
def store_line(s):
    if ssa_tot.get(s, True):
        tile = "TA::Tensor<TA::ArenaTensor<double>>"
        return (f"  TA::foreach_inplace({name(s)}, [](" + tile + "& t){{ "
                f"t = TA::detail::arena_compact<" + tile + ">(t); return t.norm(); }});\n"
                f"  cache.{name(s)} = {name(s)};\n")
    return f"  cache.{name(s)} = {name(s)};\n"
store = "".join(store_line(s) for s in sorted(boundary))
load = "".join(f"  const {ctype(s)}& {name(s)} = cache.{name(s)};\n" for s in sorted(boundary))

ti_used = set()
for s in stmts:
    if not s["td"]:
        for nm, sid in s["refs"]:
            if sid is None and nm in leaf_names: ti_used.add(nm)
pdecl = lambda p: f"const {p[1]}& {p[0]}"
prec_params = ", ".join(pdecl(p) for p in params if p[0] in ti_used)
all_params = ", ".join(pdecl(p) for p in params)

# Single-scope benchmark function: keep the t-independent block as locals
# (persistent cache, alive for the whole run — no cross-function-boundary
# struct, which triggered an ArenaTensor cross-thread lifetime race), then
# run the t-dependent residual as a lambda in a warmup + timed loop.
os.makedirs(os.path.dirname(OUT), exist_ok=True)
OUT_H = OUT.replace(".cpp", ".h")
with open(OUT_H, "w", encoding="utf-8") as f:
    f.write("#ifndef T2_SPLIT_H\n#define T2_SPLIT_H\n#include <tiledarray.h>\n#include \"ta_tensors.h\"\n")
    f.write("// Runs t-indep precompute once, then times `ntrials` t-dependent\n")
    f.write("// residuals; prints one CSV line per trial. Returns final nnz.\n")
    f.write(f"long run_warm_t2_bench(TA::World& world, {all_params}, int ntrials, int warmup);\n")
    f.write("#endif\n")
prec_body = "\n".join("  " + l.strip() for l in prec)
resid_body = "\n".join("    " + l.strip() for l in resid)
with open(OUT, "w", encoding="utf-8") as f:
    f.write('#include <tiledarray.h>\n#include <TiledArray/expressions/einsum.h>\n')
    f.write('#include <chrono>\n#include <iostream>\n#include <cmath>\n')
    f.write('#include "t2_split.h"\n#include "ta_dumper.h"\n\n')
    f.write(f"long run_warm_t2_bench(TA::World& world, {all_params}, int ntrials, int warmup) {{\n")
    f.write("  // --- t-independent block (persistent, computed once) ---\n")
    f.write(prec_body + "\n")
    f.write("  world.gop.fence();\n")
    # Make the persistent t-independent ToT arrays self-owning (single arena
    # page per tile) so their inner ArenaTensor cells don't dangle when read
    # by the multithreaded residual after this phase boundary (MPQC's
    # compact_csv_coeffs, csv.ipp). Only the boundary ToT locals are read later.
    # arena_compact only applies to the ArenaTensor build; the owning
    # Tensor<Tensor<double>> build needs no compaction (cells own their data).
    tile_t = "TA::Tensor<TA::ArenaTensor<double>>"
    f.write("#ifndef SPTC_OWNING_TOT\n")
    for s in sorted(boundary):
        if ssa_tot.get(s, True):
            f.write(f"  TA::foreach_inplace({name(s)}, []({tile_t}& t){{ "
                    f"t = TA::detail::arena_compact<{tile_t}>(t); return t.norm(); }});\n")
    f.write("#endif\n")
    f.write("  world.gop.fence();\n")
    f.write(f"  auto residual = [&]() -> {TOT} {{\n")
    f.write(resid_body + "\n")
    f.write(f"    return {name(ret_ssa)};\n  }};\n")
    f.write("  if (warmup) { auto w = residual(); world.gop.fence(); (void)w; }\n")
    f.write("  long nnz = 0;\n")
    f.write("  for (int t = 1; t <= ntrials; ++t) {\n")
    f.write("    world.gop.fence();\n")
    f.write("    auto t0 = std::chrono::high_resolution_clock::now();\n")
    f.write("    auto res = residual();\n")
    f.write("    world.gop.fence();\n")
    f.write("    auto t1 = std::chrono::high_resolution_clock::now();\n")
    f.write("    double wall = std::chrono::duration<double>(t1 - t0).count();\n")
    f.write("    auto cs = ta_compute_checksum(world, res);\n    nnz = cs.nnz;\n")
    f.write("    if (world.rank() == 0)\n")
    f.write('      std::cout << "warm_t2_residual,trial" << t << ",wall_s=" << wall\n')
    f.write('                << ",nnz=" << cs.nnz << ",sum=" << cs.sum\n')
    f.write('                << ",max_abs=" << cs.max_abs << "\\n";\n')
    f.write("  }\n  return nnz;\n}\n")

print(f"wrote {OUT}")
print(f"precompute (t-indep) stmts: {sum(1 for s in stmts if not s['td'])}")
print(f"residual (t-dep) stmts:     {sum(1 for s in stmts if s['td'])}")
print(f"boundary cached intermediates: {len(boundary)}")
print(f"precompute leaves: {sorted(ti_used)}")
