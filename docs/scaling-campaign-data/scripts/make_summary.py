#!/usr/bin/env python3
"""Regenerate SUMMARY.md from comparison.csv + results.csv + results_warm.csv."""
import csv, statistics, os
W="/proj/perf-model-gpu-PG0/jianjian-scaling"
MOLS=["C2H6","C3H8","C4H10","C5H12","C6H14"]

def load(p):
    try: return list(csv.DictReader(open(p)))
    except FileNotFoundError: return []

comp=load(W+"/comparison.csv")
cold=load(W+"/results.csv")        # repro cold raw: molecule,np,side,trial,T1_s,T2_s,...
warm=load(W+"/results_warm.csv")   # repro warm raw: molecule,np,side,trial,T2_s,...

# MPQC cold/warm T2 by molecule (np=1)
mpqc={}
for r in comp:
    if r.get('side')=='mpqc' and r.get('np')=='1':
        mpqc.setdefault(r['molecule'],{})[r['mode']]=r['T2_s']

def med_by_np(rows, key='T2_s', side=None):
    d={}
    for r in rows:
        if side and r.get('side')!=side: continue
        try:
            v=float(r[key]); n=int(r['np'])
        except (ValueError,KeyError,TypeError): continue
        d.setdefault((r['molecule'],n),[]).append(v)
    return {k:statistics.median(v) for k,v in d.items() if v}

cold_t2=med_by_np(cold)                 # repro cold T2 median
warm_t2=med_by_np(warm)                 # repro warm T2 median (side repro_warm)

NPS=[1,2,4,8,16]
out=["# Scaling campaign — running summary","",
     "## MPQC T2 wall-time (single rank, PaRSEC)","",
     "| molecule | cold (occ1) | warm (occ2) |","|---|---|---|"]
for m in MOLS:
    if m in mpqc: out.append(f"| {m} | {mpqc[m].get('cold','-')} | {mpqc[m].get('warm','-')} |")

out+=["","## Repro WARM T2 (t-dependent only) vs MPQC warm — the fair comparison","",
      "| molecule | "+" | ".join(f"np{n}" for n in NPS)+" | MPQC warm | ratio(np1) |","|---|"+"---|"*(len(NPS)+2)]
for m in MOLS:
    cells=[f"{warm_t2.get((m,n)):.2f}" if (m,n) in warm_t2 else "-" for n in NPS]
    mw=mpqc.get(m,{}).get('warm','-')
    try: ratio=f"{warm_t2[(m,1)]/float(mw):.2f}x"
    except: ratio="-"
    if any(c!="-" for c in cells): out.append(f"| {m} | "+" | ".join(cells)+f" | {mw} | {ratio} |")

out+=["","## Repro COLD T2 multi-rank scaling","",
      "| molecule | "+" | ".join(f"np{n}" for n in NPS)+" | speedup |","|---|"+"---|"*(len(NPS)+1)]
for m in MOLS:
    cells=[f"{cold_t2.get((m,n)):.1f}" if (m,n) in cold_t2 else "-" for n in NPS]
    try: sp=f"{cold_t2[(m,1)]/cold_t2[(m,16)]:.1f}x"
    except: sp="-"
    if any(c!="-" for c in cells): out.append(f"| {m} | "+" | ".join(cells)+f" | {sp} |")

out+=["","## Findings",
      "- Warm (steady-state) repro ~ MPQC warm (C2H6 1.3x); warm barely rank-scales (work too small).",
      "- Cold gap (~10x np1) is one DF half-transform; multi-rank scales it (C2H6 5.4x over 16 ranks).",
      "- proto=100 generator extent avoids the giant intermediate (3x cold np1) but t-dep value differs ~7% (correctness open).",
      "- Same TiledArray fork cd53bd3 + same SeQuant derivation both sides; difference is evaluation/tiling.",
      "- Hexane single-rank MPQC OOMs (63GB) — needs multi-rank; repro uses alkanes-v3 hexane leaves."]
open(W+"/SUMMARY.md","w").write("\n".join(out)+"\n")
print("SUMMARY.md regenerated")
