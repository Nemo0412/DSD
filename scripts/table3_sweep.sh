#!/bin/bash
# Small-scale Table 3 on NYULAB, <=3 GPUs (context.md): 1 target GPU + 2 draft
# GPUs; N > 2 drafters share draft GPUs (tier T2, see plan.md §7).
#
#   scripts/table3_sweep.sh [OUT] [METHODS] [NS]
#   scripts/table3_sweep.sh runs/table3 "vanilla specpp eagle3" "1 2 4 7"
#
# specpp needs a per-position acceptance profile; it is calibrated first with
# a static gamma=8 run at N=1 (reused if already present).
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env_nyulab.sh
OUT=${1:-runs/table3}; METHODS=${2:-vanilla specpp}; NS=${3:-1 2 4 7}
PER_DRAFT_RPS=${PER_DRAFT_RPS:-1.0}; RTT_MS=${RTT_MS:-10}; GAMMA=${GAMMA:-4}
# client: dsdSim's scenario -- requests are born at the drafters' clients
# (prompt uplink + token downlink of RTT/2 each); datacenter: they arrive at
# the target.
SCENARIO=${SCENARIO:-client}
scen_args=(); [ "$SCENARIO" = client ] && scen_args=(--client-side)
mkdir -p "$OUT"
# Shared box: GPUs are re-checked before every run (another user's job can
# come back mid-sweep -- seen on NYULAB), and after every run we record
# whether anyone else ran on our GPUs meanwhile (gpu_check.txt).
pick_gpus() {
    until GPUS=$(scripts/free_gpus.sh 3 2>/dev/null); do sleep 120; done
    IFS=, read -r T D1 D2 <<<"$GPUS"
}
others_on() {  # gpu list -> lines "gpu_bus pid user" for processes not ours
    local ids=$1
    for bus in $(nvidia-smi -i "$ids" --query-gpu=pci.bus_id --format=csv,noheader); do
        nvidia-smi --query-compute-apps=gpu_bus_id,pid --format=csv,noheader | grep -i "$bus" \
            | while IFS=', ' read -r b pid; do u=$(ps -o user= -p "$pid"); [ "$u" != "$USER" ] && echo "$b $pid $u"; done
    done
}
pick_gpus
echo "target GPU $T, draft GPUs $D1,$D2 -> $OUT (per_draft_rps=$PER_DRAFT_RPS rtt=$RTT_MS)"
common() {  # N
    local n=$1 nreq=$(( 120 * $1 )); (( nreq > 400 )) && nreq=400
    echo --n-requests $nreq --per-draft-rps "$PER_DRAFT_RPS" --rtt-ms "$RTT_MS" --chat-template --no-prefix-caching "${scen_args[@]}"
}

if [[ " $METHODS " == *" specpp "* ]] && [ ! -f "$OUT/calib_g8/profile.json" ]; then
    DSD_DRAFT_KIND=vllm scripts/run_nto1.sh "$OUT/calib_g8" 1 "$T" "$D1" $(common 1) --gamma 8 > "$OUT/calib_g8.log" 2>&1 \
        || { echo "calibration failed"; tail -20 "$OUT/calib_g8.log"; exit 1; }
    $DSD_PY scripts/build_acceptance_profile.py "$OUT/calib_g8" -o "$OUT/calib_g8/profile.json" --edges 64,96
fi

for m in $METHODS; do
    for n in $NS; do
        run="$OUT/${m}_N$n"
        [ -f "$run/dsd_metrics.json" ] && { echo "skip $run (done)"; continue; }
        case $m in
            vanilla) extra=(--gamma "$GAMMA") ;;
            specpp)  extra=(--gamma 8 --gamma-policy specpp --acceptance-profile "$OUT/calib_g8/profile.json") ;;
            eagle3)  extra=(--gamma "$GAMMA" --draft-kind eagle3) ;;
            *) echo "unknown method $m"; exit 1 ;;
        esac
        pick_gpus
        echo "== $m N=$n (target GPU $T, draft GPUs $D1,$D2)"
        kind=vllm; model=${DSD_DRAFT_MODEL:-meta-llama/Llama-3.2-1B-Instruct}
        [ "$m" = eagle3 ] && kind=eagle3 && model=${EAGLE3_HEAD:-yuhuili/EAGLE3-LLaMA3.1-Instruct-8B}
        DSD_DRAFT_KIND=$kind DSD_DRAFT_MODEL=$model \
        scripts/run_nto1.sh "$run" "$n" "$T" "$D1,$D2" $(common "$n") "${extra[@]}" > "$run.log" 2>&1 \
            || { echo "$run failed"; tail -20 "$run.log"; }
        o=$(others_on "$T,$D1,$D2")
        { echo "gpus $T,$D1,$D2"; [ -n "$o" ] && echo "OTHER PROCESSES AT END (timings suspect):" && echo "$o" || echo "clean"; } \
            > "$run/gpu_check.txt" 2>/dev/null
    done
done
$DSD_PY scripts/make_table3.py "$OUT" | tee "$OUT/table3.md"
$DSD_PY scripts/deployment_report.py "$OUT" > "$OUT/deployment.md"
