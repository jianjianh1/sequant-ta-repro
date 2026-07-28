#!/bin/bash
# Verification experiment for docs/MPQC_EVALUATION.md §8 (the cold-gap lever):
# does the repro's OCC TILING / pmap of the DF-carrying ToT arrays actually
# move the cold np=16 residual time? MPQC's residual ToT inherits the CSV
# energies array's trange/shape/pmap (cck.ipp:1563-1565); the repro builds its
# own tiling in build_tot_array. Sweep the occ tiling on C4H10 (active occ
# extent = 17) cold at np=16 and compare whole-T2 wall time across configs.
#
# NB: the committed §11 sweep used SPTC_COARSE_OCC=9 (ethane-specific); for
# C4H10 that never matches extent 17, so the occ dims silently fell back to
# adaptive (17/8=2). This experiment sets COARSE_OCC=17 correctly.
set -uo pipefail
W=/proj/perf-model-gpu-PG0/jianjian-scaling
BIN=$W/bin/ta_sequant_native_residual_main
MOL=C4H10
LEAFDIR=$W/leaves/${MOL}_coo
OCC_EXT=17
NP=16
TRIALS=2
OUT=$W/occ_tiling_experiment.csv
LOGDIR=$W/logs
HF=$W/hosts/np${NP}.txt

nodes=$(awk '{print $1}' "$HF")

echo "[$(date +%T)] staging $LEAFDIR -> /local on $NP nodes"
for n in $nodes; do
  ssh -o ConnectTimeout=10 "$n" "rm -rf /local/jianjian/* && mkdir -p /local/jianjian/$MOL" </dev/null
  scp -q -p "$LEAFDIR"/*.txt "$n:/local/jianjian/$MOL/" </dev/null
done
echo "[$(date +%T)] staged"

echo "config,trial,T1_s,T2_s,t2_nnz,t2_sum,t2_sumsq" > "$OUT"

run_cfg () {
  local name=$1; shift
  echo "[$(date +%T)] === config: $name ($*) ==="
  local out
  out=$(timeout 1500 mpirun --hostfile "$HF" -np "$NP" --bind-to none \
      --mca btl_tcp_if_include 10.10.1.0/24 -x LD_LIBRARY_PATH="$W/lib" \
      "$@" \
      -x SPTC_MAD_WAIT_POLICY=yield -x MAD_NUM_THREADS=8 \
      -x SPTC_TRIALS="$TRIALS" -x SPTC_WARMUP=1 \
      "$BIN" "/local/jianjian/$MOL" 2>&1)
  echo "$out" > "$LOGDIR/occexp_${name}.log"
  echo "$out" | awk -F, -v C="$name" '
    $2=="whole_t1_residual" && $3=="whole_residual" {t1[$4]=$6}
    $2=="whole_t2_residual" && $3=="whole_residual" {
       print C","$4","t1[$4]","$6","$10","$11","$12 }' | sort -u >> "$OUT"
  echo "[$(date +%T)] $name done"
}

# 1. Baseline = the committed §11 config (COARSE_OCC=9 does NOT apply to C4H10,
#    so occ dims are adaptive ~2). This reproduces the §11 cold C4H10 np=16 number.
run_cfg baseline_committed -x SPTC_COARSE_OCC=9  -x SPTC_OCC_TILE=2 -x SPTC_COARSE_PAD=0 -x SPTC_TILES_PER_DIM=8
# 2. Correct coarse occ, tile 2 (finest — closest to MPQC's per-pair-ish tiling)
run_cfg occ17_t2       -x SPTC_COARSE_OCC=17 -x SPTC_OCC_TILE=2 -x SPTC_COARSE_PAD=0 -x SPTC_TILES_PER_DIM=8
# 3. Correct coarse occ, tile 4
run_cfg occ17_t4       -x SPTC_COARSE_OCC=17 -x SPTC_OCC_TILE=4 -x SPTC_COARSE_PAD=0 -x SPTC_TILES_PER_DIM=8
# 4. Correct coarse occ, tile 8 (coarsest)
run_cfg occ17_t8       -x SPTC_COARSE_OCC=17 -x SPTC_OCC_TILE=8 -x SPTC_COARSE_PAD=0 -x SPTC_TILES_PER_DIM=8
# 5. No coarse-occ override at all (pure adaptive TPD=8 for every dim)
run_cfg noocc_tpd8                                                                    -x SPTC_TILES_PER_DIM=8
# 6. Coarser everything (TPD=6, the warm optimum) + correct occ
run_cfg occ17_t4_tpd6  -x SPTC_COARSE_OCC=17 -x SPTC_OCC_TILE=4 -x SPTC_COARSE_PAD=0 -x SPTC_TILES_PER_DIM=6

echo "[$(date +%T)] cleaning /local"
for n in $nodes; do ssh -o ConnectTimeout=10 "$n" "rm -rf /local/jianjian/$MOL" </dev/null; done
echo "[$(date +%T)] EXPERIMENT COMPLETE"
column -t -s, "$OUT"
