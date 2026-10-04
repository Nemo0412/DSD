#!/bin/bash
# One N:M run: N drafter servers on DRAFT_GPUS, M target servers (one per
# TARGET_GPUS entry, each fanning out to every drafter), a router frontend
# choosing a target per request.
#
#   scripts/run_nm.sh OUT N TARGET_GPUS DRAFT_GPUS POLICY [target args...] -- [workload args...]
#   scripts/run_nm.sh runs/nm/jsq 4 0,1 2 jsq --gamma 4 --rtt-ms 10 -- --per-draft-rps 1 --n-requests 400 --chat-template
set -uo pipefail
OUT=${1:?OUT}; N=${2:?N}; TGPUS=${3:?TARGET_GPUS}; DGPUS=${4:?DRAFT_GPUS}; POLICY=${5:?POLICY}; shift 5
targs=(); wargs=(); seen=0
for a in "$@"; do if [ "$a" = -- ]; then seen=1; elif [ $seen = 0 ]; then targs+=("$a"); else wargs+=("$a"); fi; done
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${DSD_PY:-python}
DRAFT_MODEL=${DSD_DRAFT_MODEL:-meta-llama/Llama-3.2-1B-Instruct}
TARGET_MODEL=${DSD_TARGET_MODEL:-meta-llama/Llama-3.1-8B-Instruct}
BUDGET=${DSD_DRAFT_GPU_BUDGET:-0.8}
mkdir -p "$OUT"
IFS=, read -r -a dg <<<"$DGPUS"; IFS=, read -r -a tg <<<"$TGPUS"
per_gpu=$(( (N + ${#dg[@]} - 1) / ${#dg[@]} ))
frac=$(python3 -c "print(round($BUDGET / $per_gpu, 3))")
pids=(); cleanup() { for p in "${pids[@]}"; do kill "$p" 2>/dev/null; done; wait 2>/dev/null; }; trap cleanup EXIT
wait_ready() {  # file pid log
    for _ in $(seq 1 300); do [ -f "$1" ] && return 0; kill -0 "$2" 2>/dev/null || { echo "died: $3"; tail -30 "$3"; exit 1; }; sleep 2; done
    echo "not ready: $3"; exit 1
}

daddrs=()
for ((i = 0; i < N; i++)); do
    r="$OUT/draft_d$i.ready"; rm -f "$r"
    CUDA_VISIBLE_DEVICES=${dg[$((i % ${#dg[@]}))]} "$PY" "$ROOT/scripts/run_draft_server.py" --model "$DRAFT_MODEL" \
        --drafter-id "d$i" --bind "tcp://127.0.0.1:0" --gpu-frac "$frac" --dtype bfloat16 \
        --max-model-len 2048 --ready-file "$r" > "$OUT/draft_d$i.log" 2>&1 &
    pids+=($!); wait_ready "$r" "$!" "$OUT/draft_d$i.log"   # one at a time: KV sizing races otherwise
    daddrs+=(--draft-addr "d$i=$(cat "$r")")
done
taddrs=()
for ((j = 0; j < ${#tg[@]}; j++)); do
    r="$OUT/target_t$j.ready"; rm -f "$r"
    CUDA_VISIBLE_DEVICES=${tg[$j]} "$PY" -m dsdreal.router.target_server --target-model "$TARGET_MODEL" \
        --bind "tcp://127.0.0.1:0" --ready-file "$r" "${daddrs[@]}" "${targs[@]}" \
        > "$OUT/target_t$j.log" 2>&1 &
    pids+=($!); wait_ready "$r" "$!" "$OUT/target_t$j.log"   # sequential, same reason
    taddrs+=(--target-addr "t$j=$(cat "$r")")
done
echo "N=$N drafters, M=${#tg[@]} targets ready; policy=$POLICY"
"$PY" -m dsdreal.router.frontend "${taddrs[@]}" --policy "$POLICY" --target-model "$TARGET_MODEL" \
    --n-drafters "$N" --out "$OUT" "${wargs[@]}" 2>&1 | tee "$OUT/router.log" | grep -v "WARNING"
rc=${PIPESTATUS[0]}; echo "router exit $rc"; exit $rc
