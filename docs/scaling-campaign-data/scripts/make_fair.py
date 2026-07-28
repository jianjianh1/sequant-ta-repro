#!/usr/bin/env python3
"""Fair same-np repro-vs-MPQC tables (T2 wall-time), warm and cold.
Repro: results_warm.csv / results.csv. MPQC: mpqc_mr.csv (np>=4) + comparison.csv (np=1)."""
import csv, statistics
W="/proj/perf-model-gpu-PG0/jianjian-scaling"
MOLS=["C2H6","C3H8","C4H10","C5H12"]; NPS=[1,2,4,8,16]
def load(p):
    try: return list(csv.DictReader(open(p)))
    except FileNotFoundError: return []

def repro_med(rows,key):
    d={}
    for r in rows:
        try: v=float(r[key]); n=int(r['np'])
        except: continue
        d.setdefault((r['molecule'],n),[]).append(v)
    return {k:statistics.median(v) for k,v in d.items() if v}
rw=repro_med(load(W+"/results_warm.csv"),'T2_s')   # repro warm
rc=repro_med(load(W+"/results.csv"),'T2_s')         # repro cold

# MPQC: np=1 from comparison.csv, np>=4 from mpqc_mr.csv
mq={'warm':{},'cold':{}}
for r in load(W+"/comparison.csv"):
    if r.get('side')=='mpqc':
        try: mq[r['mode']][(r['molecule'],1)]=float(r['T2_s'])
        except: pass
for r in load(W+"/mpqc_mr.csv"):
    if r['mode'] in ('warm','cold'):
        try: mq[r['mode']][(r['molecule'],int(r['np']))]=float(r['T2_s'])
        except: pass

def tbl(repro,mpqc,label):
    out=[f"**{label} T2 (s) — repro / MPQC / ratio, same np:**","",
         "| mol | metric | "+" | ".join(f"np{n}" for n in NPS)+" |","|---|---|"+"---|"*len(NPS)]
    for m in MOLS:
        rr=[f"{repro.get((m,n)):.1f}" if (m,n) in repro else "-" for n in NPS]
        mm=[f"{mpqc.get((m,n)):.1f}" if (m,n) in mpqc else "-" for n in NPS]
        rt=[f"{repro[(m,n)]/mpqc[(m,n)]:.1f}x" if (m,n) in repro and (m,n) in mpqc else "-" for n in NPS]
        out.append(f"| {m} | repro | "+" | ".join(rr)+" |")
        out.append(f"| {m} | MPQC | "+" | ".join(mm)+" |")
        out.append(f"| {m} | ratio | "+" | ".join(rt)+" |")
    return "\n".join(out)
print(tbl(rw,mq['warm'],"WARM")); print(); print(tbl(rc,mq['cold'],"COLD"))
