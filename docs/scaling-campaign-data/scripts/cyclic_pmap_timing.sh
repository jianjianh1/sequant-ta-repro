#!/bin/bash
# Timing payoff test for MPQC_EVALUATION.md lever (a): does SPTC_CYCLIC_PMAP
# (cyclic pmap that eliminates the idle ranks) actually speed up cold np=16?
# C4H10, owning-ToT binary, committed §11 config, cyclic OFF vs ON, np=16 on
# nodes 16-31. A pmap change is checksum-invariant, so the T2 sum must match
# between the two (correctness guard) while T2 wall time is the comparison.
set -uo pipefail
W=/proj/perf-model-gpu-PG0/jianjian-scaling
BIN=$W/bin/ta_cyclic_owning
MOL=C4H10
LEAFDIR=$W/leaves/${MOL}_coo
NP=16
TRIALS=3
HF=$W/hosts/np${NP}.txt
OUT=$W/cyclic_pmap_timing.csv
nodes=$(awk '{print $1}' "$HF")

echo "[$(date +%T)] staging $LEAFDIR -> /local on $NP nodes"
for n in $nodes; do
  ssh -o ConnectTimeout=10 "$n" "rm -rf /local/jianjian/* && mkdir -p /local/jianjian/$MOL" </dev/null
  scp -q -p "$LEAFDIR"/*.txt "$n:/local/jianjian/$MOL/" </dev/null
done
echo "[$(date +%T)] staged"

echo "config,trial,T1_s,T2_s,t2_nnz,t2_sum,t2_sumsq" > "$OUT"
run () {
  local name=$1; shift
  echo "[$(date +%T)] === $name ($*) ==="
  local out
  out=$(timeout 1500 mpirun --hostfile "$HF" -np "$NP" --bind-to none \
      --mca btl_tcp_if_include 10.10.1.0/24 -x LD_LIBRARY_PATH="$W/lib" \
      "$@" -x SPTC_COARSE_OCC=9 -x SPTC_OCC_TILE=2 -x SPTC_COARSE_PAD=0 -x SPTC_TILES_PER_DIM=8 \
      -x SPTC_MAD_WAIT_POLICY=yield -x MAD_NUM_THREADS=8 -x SPTC_TRIALS="$TRIALS" -x SPTC_WARMUP=1 \
      "$BIN" "/local/jianjian/$MOL" 2>&1)
  echo "$out" > "$W/logs/cyclic_${name}.log"
  echo "$out" | awk -F, -v C="$name" '
    $2=="whole_t1_residual" && $3=="whole_residual" {t1[$4]=$6}
    $2=="whole_t2_residual" && $3=="whole_residual" {
       print C","$4","t1[$4]","$6","$10","$11","$12 }' | sort -u | tee -a "$OUT"
  echo "[$(date +%T)] $name done"
}

run pmap_default                          # cyclic OFF = the committed §11 path
run pmap_cyclic -x SPTC_CYCLIC_PMAP=1     # cyclic ON

echo "[$(date +%T)] cleaning /local"
for n in $nodes; do ssh -o ConnectTimeout=10 "$n" "rm -rf /local/jianjian/$MOL" </dev/null; done
echo "[$(date +%T)] DONE"
column -t -s, "$OUT"
