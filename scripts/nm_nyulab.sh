#!/bin/bash
# N:M on NYULAB within the 3-GPU limit: 2 targets (one GPU each) + 2 drafters
# sharing the third GPU, vanilla gamma=4, RR vs JSQ at the same offered load.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env_nyulab.sh
OUT=${1:-runs/nm}; RPS=${RPS:-4}
mkdir -p "$OUT"
until GPUS=$(scripts/free_gpus.sh 3 2>/dev/null); do sleep 120; done
IFS=, read -r T1 T2 D <<<"$GPUS"
for pol in rr jsq; do
    [ -f "$OUT/$pol/dsd_metrics.json" ] && continue
    scripts/run_nm.sh "$OUT/$pol" 2 "$T1,$T2" "$D" "$pol" --gamma 4 --rtt-ms 10 --no-prefix-caching -- \
        --per-draft-rps "$RPS" --n-requests 400 --chat-template > "$OUT/$pol.log" 2>&1
    echo "$pol rc=$?" >> "$OUT/status"
done
echo done >> "$OUT/status"
