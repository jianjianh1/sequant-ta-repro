#!/bin/bash
# MPQC multi-rank sweep for the fair same-np comparison. Runs MPQC (PaRSEC)
# across node16-31 for each molecule x np, parses WholeResidualWallTime
# (occ1=cold, occ2=warm, R=2=T2). Idempotent (skips molecule,np already in
# mpqc_mr.csv). np=1 single-rank is already in comparison.csv.
set -uo pipefail
W=/proj/perf-model-gpu-PG0/jianjian-scaling
SIF=$W/mpqc-latest.sif
CSV=$W/mpqc_mr.csv
cd /users/jianjian/mpqc-benchmark
[[ -f $CSV ]] || echo "molecule,np,mode,T1_s,T2_s,note" > "$CSV"
log(){ echo "[$(date '+%m-%d %T')] $*" | tee -a "$W/logs/mpqc_mr_sweep.log"; }
log "=== mpqc_mr_sweep START ==="
for M in C4H10 C5H12 C3H8 C2H6; do
  for N in 16 8 4; do
    if grep -q "^$M,$N," "$CSV" 2>/dev/null; then log "$M np=$N present, skip"; continue; fi
    L=$W/logs/mpqc_mr_${M}_np${N}.log
    log "MPQC $M np=$N"
    timeout 2400 sudo bin/run-mpqc-mpirun.sh "$SIF" "$W/inputs/${M}-perf.json" \
      -n "$N" -H "$W/hosts/np$N.txt" --log "$L" >/dev/null 2>&1
    python3 - "$L" "$M" "$N" "$CSV" <<'PY'
import sys,re
log,M,N,CSV=sys.argv[1:5]
w={'R=1':[],'R=2':[]}
for ln in open(log,errors='ignore'):
    m=re.search(r'WholeResidualWallTime \| (\d+)ns \| (R=\d)',ln)
    if m: w[m.group(2)].append(int(m.group(1))/1e9)
def g(k,i): return f"{w[k][i]:.2f}" if len(w[k])>i else "-"
with open(CSV,'a') as f:
    if len(w['R=2'])>=2:
        f.write(f"{M},{N},cold,{g('R=1',0)},{g('R=2',0)},occ1\n")
        f.write(f"{M},{N},warm,{g('R=1',1)},{g('R=2',1)},occ2\n")
    else:
        f.write(f"{M},{N},FAIL,-,-,insufficient WallTime lines\n")
PY
    log "$M np=$N: $(grep "^$M,$N," "$CSV" | head -1)"
  done
done
log "=== mpqc_mr_sweep COMPLETE ==="
