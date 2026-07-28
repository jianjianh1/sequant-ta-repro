#!/bin/bash
# Repro rank-sweep driver for one molecule across node16-31 (1 rank/node).
#
# Usage: repro_sweep.sh <MOL> <LEAFDIR_on_proj> [NP_LIST] [TRIALS]
#   MOL       label (e.g. C2H6)
#   LEAFDIR   sptc_coo dir on shared /proj (staged to /local per node)
#   NP_LIST   space-separated rank counts (default "1 2 4 8 16")
#   TRIALS    timed trials per point (default 3)
#
# Stages LEAFDIR to /local/jianjian/<MOL> on every node used, runs the locked
# repro config with --bind-to none (each 1-rank/node uses the whole node),
# appends timing+checksum rows to results.csv, then cleans /local.
set -uo pipefail
W=/proj/perf-model-gpu-PG0/jianjian-scaling
BIN=$W/bin/ta_sequant_native_residual_main
CSV=$W/results.csv
MOL=${1:?MOL}; LEAFDIR=${2:?LEAFDIR}; NP_LIST=${3:-"1 2 4 8 16"}; TRIALS=${4:-3}
ALLNODES=$(grep -oE '\bnode(1[6-9]|2[0-9]|3[01])\b' /etc/hosts | sort -uV)
maxN=$(echo $NP_LIST | tr ' ' '\n' | sort -n | tail -1)
stage_nodes=$(echo "$ALLNODES" | head -n "$maxN")

echo "[$(date +%T)] $MOL: staging $LEAFDIR -> /local/jianjian/$MOL on $maxN nodes"
for n in $stage_nodes; do
  ssh -o ConnectTimeout=10 "$n" "rm -rf /local/jianjian/* && mkdir -p /local/jianjian/$MOL" </dev/null
  scp -q -p "$LEAFDIR"/*.txt "$n:/local/jianjian/$MOL/" </dev/null
done
echo "[$(date +%T)] $MOL: staged"

[[ -f $CSV ]] || echo "molecule,np,side,trial,T1_s,T2_s,t1_nnz,t1_sum,t2_nnz,t2_sum,t2_sumsq,note" > "$CSV"

for N in $NP_LIST; do
  echo "[$(date +%T)] $MOL np=$N: running ($TRIALS trials)"
  out=$(timeout 1500 mpirun --hostfile "$W/hosts/np$N.txt" -np "$N" --bind-to none \
      --mca btl_tcp_if_include 10.10.1.0/24 -x LD_LIBRARY_PATH="$W/lib" \
      -x SPTC_COARSE_OCC=9 -x SPTC_OCC_TILE=2 -x SPTC_COARSE_PAD=0 -x SPTC_TILES_PER_DIM=8 \
      -x SPTC_MAD_WAIT_POLICY=yield -x MAD_NUM_THREADS=8 -x SPTC_TRIALS="$TRIALS" -x SPTC_WARMUP=1 \
      "$BIN" "/local/jianjian/$MOL" 2>&1)
  echo "$out" > "$W/logs/repro_${MOL}_np${N}.log"
  # rank-0 emits CSV-style rows: mol,equation,stage,trial,nranks,wall_s,...,nnz,sum,sumsq,max_abs,note
  echo "$out" | awk -F, -v M="$MOL" -v N="$N" '
    $2=="whole_t1_residual" && $3=="whole_residual" {t1[$4]=$6; n1=$10; s1=$11}
    $2=="whole_t2_residual" && $3=="whole_residual" {t2[$4]=$6; n2=$10; s2=$11; q2=$12;
       print M","N",repro,"$4","t1[$4]","$6","n1","s1","n2","s2","q2",ok"}
  ' | sort -u >> "$CSV"
  echo "[$(date +%T)] $MOL np=$N: done -> $(grep -c "^$MOL,$N,repro" "$CSV") rows"
done

echo "[$(date +%T)] $MOL: cleaning /local"
for n in $stage_nodes; do ssh -o ConnectTimeout=10 "$n" "rm -rf /local/jianjian/$MOL" </dev/null; done
echo "[$(date +%T)] $MOL: SWEEP COMPLETE"
