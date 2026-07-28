#!/usr/bin/env python3
"""Derive per-molecule MPQC ref/perf input JSONs from alkanes-v3 *-full.json.

ref  : keep sequant.trace on + leaf dump enabled, retarget tns_outdir to a
       fresh per-molecule dir under jianjian-scaling/leaves/<MOL>  (one run
       yields WholeResidualChecksum AND fresh leaves the repro consumes).
perf : disable the leaf dump and per-term eval trace for clean
       WholeResidualWallTime timing (trace-independent).
Stdlib only. Usage: make_mpqc_variants.py <name-full.json> <MOL> <outdir> <leafroot>
"""
import json, sys, os

src, mol, outdir, leafroot = sys.argv[1:5]
cfg = json.load(open(src))
tr = cfg["wfn"]["sequant"]["trace"]

# The template's molecule.file_name is relative to the template's own dir;
# absolutize it so the variant works from its new location.
xyz = cfg["molecule"]["file_name"]
if not os.path.isabs(xyz):
    xyz = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(src)), xyz))
cfg["molecule"]["file_name"] = xyz

# ref: dump leaves fresh + trace on
ref = json.loads(json.dumps(cfg))
rtr = ref["wfn"]["sequant"]["trace"]
rtr["eval"] = False            # per-term eval trace off (heavy); eval_level drives checksum
rtr["eval_level"] = 1
rtr["selected"]["enabled"] = True
rtr["selected"]["write_tns"] = True
rtr["selected"]["tns_outdir"] = os.path.join(leafroot, mol)
os.makedirs(os.path.join(leafroot, mol), exist_ok=True)

# perf: no dump, no trace
perf = json.loads(json.dumps(cfg))
ptr = perf["wfn"]["sequant"]["trace"]
ptr["eval"] = False
ptr["eval_level"] = 0
ptr["selected"]["enabled"] = False
ptr["selected"]["write_tns"] = False

os.makedirs(outdir, exist_ok=True)
for tag, obj in (("ref", ref), ("perf", perf)):
    p = os.path.join(outdir, f"{mol}-{tag}.json")
    json.dump(obj, open(p, "w"), indent=1)
    print("wrote", p)
