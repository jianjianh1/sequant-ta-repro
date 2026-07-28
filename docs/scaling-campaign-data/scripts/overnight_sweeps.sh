#!/bin/bash
# Lane B overnight orchestrator (node16-31): warm sweeps (all 5) then cold
# sweeps (4 bigger; C2H6 cold already has 3-trial data). Idempotent: skips any
# molecule already present in the target CSV. Checkpoints to /proj throughout.
set -uo pipefail
W=/proj/perf-model-gpu-PG0/jianjian-scaling
DA=/proj/perf-model-gpu-PG0/jianjian-alkanes-v3
log(){ echo "[$(date '+%m-%d %T')] $*" | tee -a "$W/logs/overnight.log"; }

# molecule -> leaf dir (C6H14 uses alkanes-v3; its fresh MPQC dump OOM'd single-rank)
leafdir(){ case "$1" in C6H14) echo "$DA/sptc_coo/C6H14";; *) echo "$W/leaves/$1_coo";; esac; }

MOLS="C2H6 C3H8 C4H10 C5H12 C6H14"

log "=== overnight_sweeps START ==="

# Stage 1: WARM sweeps (the fair steady-state comparison) — highest priority
for M in $MOLS; do
  if grep -q "^$M," "$W/results_warm.csv" 2>/dev/null; then log "warm $M present, skip"; continue; fi
  D=$(leafdir "$M")
  if [ ! -e "$D/g_m_1_m_2_Κ_1.txt" ]; then log "warm $M: leaves missing at $D, skip"; continue; fi
  log "WARM sweep $M (leaves $D)"
  bash "$W/scripts/warm_sweep.sh" "$M" "$D" "16 8 4 2 1" 3 >> "$W/logs/overnight.log" 2>&1 \
    && log "WARM sweep $M done" || log "WARM sweep $M FAILED (rc=$?)"
done

# Stage 2: COLD sweeps, 1 trial (C2H6 already has 3-trial cold in results.csv)
for M in C2H6 C3H8 C4H10 C5H12 C6H14; do
  if grep -q "^$M,.*,repro," "$W/results.csv" 2>/dev/null; then log "cold $M present, skip"; continue; fi
  D=$(leafdir "$M")
  if [ ! -e "$D/g_m_1_m_2_Κ_1.txt" ]; then log "cold $M: leaves missing, skip"; continue; fi
  log "COLD sweep $M (1 trial, leaves $D)"
  bash "$W/scripts/repro_sweep.sh" "$M" "$D" "16 8 4 2 1" 1 >> "$W/logs/overnight.log" 2>&1 \
    && log "COLD sweep $M done" || log "COLD sweep $M FAILED (rc=$?)"
done

log "=== overnight_sweeps COMPLETE ==="
