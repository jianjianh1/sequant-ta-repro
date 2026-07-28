#!/bin/bash
# Warm-T2 rank-sweep for one molecule across node16-31 (1 rank/node).
# Uses ta_warm_t2 (t-independent precomputed once, only t-dependent residual
# timed) — the fair analog of MPQC's warm iter2 WholeResidualWallTime.
# Assumes leaves already staged to /local/jianjian/<MOL> by repro_sweep.sh,
# or stages them if absent. Usage: warm_sweep.sh <MOL> <LEAFDIR> [NP_LIST] [TRIALS]
set -uo pipefail
W=/proj/perf-model-gpu-PG0/jianjian-scaling
BIN=$W/bin/ta_warm_t2
CSV=$W/results_warm.csv
MOL=${1:?}; LEAFDIR=${2:?}; NP_LIST=${3:-"1 2 4 8 16"}; TRIALS=${4:-3}
ALL=$(grep -oE '\bnode(1[6-9]|2[0-9]|3[01])\b' /etc/hosts | sort -uV)
maxN=$(echo $NP_LIST | tr ' ' '\n' | sort -n | tail -1)
for n in $(echo "$ALL" | head -n "$maxN"); do
  ssh -o ConnectTimeout=10 "$n" "mkdir -p /local/jianjian/$MOL; find /local/jianjian -maxdepth 1 -mindepth 1 -type d ! -name $MOL -exec rm -rf {} + 2>/dev/null" </dev/null
  ssh -o ConnectTimeout=10 "$n" "test -e /local/jianjian/$MOL/g_m_1_m_2_Κ_1.txt" </dev/null 2>/dev/null \
    || scp -q -p "$LEAFDIR"/*.txt "$n:/local/jianjian/$MOL/" </dev/null
done
[[ -f $CSV ]] || echo "molecule,np,side,trial,T2_s,t2_nnz,t2_sum,note" > "$CSV"
for N in $NP_LIST; do
  echo "[$(date +%T)] $MOL warm np=$N"
  out=$(timeout 1500 mpirun --hostfile "$W/hosts/np$N.txt" -np "$N" --bind-to none \
      --mca btl_tcp_if_include 10.10.1.0/24 -x LD_LIBRARY_PATH="$W/lib" \
      -x SPTC_COARSE_OCC=9 -x SPTC_OCC_TILE=2 -x SPTC_COARSE_PAD=0 -x SPTC_TILES_PER_DIM=8 \
      -x SPTC_MAD_WAIT_POLICY=yield -x MAD_NUM_THREADS=8 -x SPTC_TRIALS="$TRIALS" -x SPTC_WARMUP=1 \
      "$BIN" "/local/jianjian/$MOL" 2>&1)
  echo "$out" > "$W/logs/warm_${MOL}_np${N}.log"
  echo "$out" | awk -F, -v M="$MOL" -v N="$N" '
    /^warm_t2_residual,/ {
      split($3,a,"="); split($4,b,"="); split($5,c,"=");
      print M","N",repro_warm,"$2","a[2]","b[2]","c[2]",ok" }' >> "$CSV"
done
echo "[$(date +%T)] $MOL warm sweep COMPLETE"
