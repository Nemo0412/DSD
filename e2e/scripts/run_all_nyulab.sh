#!/bin/bash
# Unattended pipeline on NYULAB: waits for free GPUs (never takes a GPU
# someone else holds, never more than 3), then runs, in order:
#   1. Step A gate (2 GPUs)   2. Table 3 sweep (3 GPUs)   3. Step D gate (2 GPUs)
# Each stage is skipped if its PASS marker / results already exist, so the
# script can simply be restarted. Progress: runs/pipeline.status
#
#   nohup scripts/run_all_nyulab.sh > runs/pipeline.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs
STATUS=runs/pipeline.status
say() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }
wait_gpus() {  # N
    local n=$1
    until scripts/free_gpus.sh "$n" > /dev/null 2>&1; do sleep 120; done
}

if ! grep -q "STEP A GATE PASS" runs/stepA.out 2>/dev/null; then
    say "waiting for 2 free GPUs (step A)"; wait_gpus 2
    say "step A gate running"
    scripts/gate_stepA.sh runs/stepA > runs/stepA.out 2>&1
    say "step A: $(tail -1 runs/stepA.out)"
fi

if [ ! -f runs/table3/table3.md ]; then
    say "waiting for 3 free GPUs (table 3)"; wait_gpus 3
    say "table 3 sweep running"
    scripts/table3_sweep.sh runs/table3 "vanilla specpp eagle3" "1 2 4 7" > runs/table3.out 2>&1
    say "table 3 sweep finished (rc=$?)"
fi

if ! grep -q "STEP D GATE PASS" runs/stepD.out 2>/dev/null; then
    say "waiting for 2 free GPUs (step D)"; wait_gpus 2
    say "step D gate running"
    scripts/gate_stepD.sh runs/stepD > runs/stepD.out 2>&1
    say "step D: $(tail -1 runs/stepD.out)"
fi
say "pipeline done"
