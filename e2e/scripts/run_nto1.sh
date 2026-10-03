#!/bin/bash
# One N:1 run on a plain multi-GPU box (no slurm): N drafter servers spread
# round-robin over the draft GPUs (several per GPU = tier T2 when N exceeds
# them), one online target driver on its own GPU.
#
#   scripts/run_nto1.sh OUT N TARGET_GPU DRAFT_GPUS [driver args...]
#   scripts/run_nto1.sh runs/t3/vanilla_N2 2 0 1,2 --gamma 4 --rtt-ms 10
#
# Env: DSD_PY (python), DSD_DRAFT_MODEL, DSD_TARGET_MODEL, DSD_DRAFT_DTYPE,
#      DSD_DRAFT_GPU_BUDGET (fraction of a draft GPU shared by its drafters),
#      DSD_DRAFT_KIND (vllm | eagle3; for eagle3 pass --draft-kind eagle3 to the driver too).
set -uo pipefail
OUT=${1:?OUT}; N=${2:?N}; TGPU=${3:?TARGET_GPU}; DGPUS=${4:?DRAFT_GPUS}; shift 4
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${DSD_PY:-python}
DRAFT_MODEL=${DSD_DRAFT_MODEL:-meta-llama/Llama-3.2-1B-Instruct}
TARGET_MODEL=${DSD_TARGET_MODEL:-meta-llama/Llama-3.1-8B-Instruct}
DRAFT_DTYPE=${DSD_DRAFT_DTYPE:-bfloat16}
BUDGET=${DSD_DRAFT_GPU_BUDGET:-0.8}
DRAFT_KIND=${DSD_DRAFT_KIND:-vllm}          # vllm | eagle3 (DSD_DRAFT_MODEL = the head)
kind_args=(--kind "$DRAFT_KIND")
[ "$DRAFT_KIND" = eagle3 ] && kind_args+=(--eagle-target-model "$TARGET_MODEL")

mkdir -p "$OUT"
IFS=, read -r -a dg <<<"$DGPUS"
per_gpu=$(( (N + ${#dg[@]} - 1) / ${#dg[@]} ))
frac=$(python3 -c "print(round($BUDGET / $per_gpu, 3))")

pids=(); addrs=()
cleanup() { for p in "${pids[@]}"; do kill "$p" 2>/dev/null; done; wait 2>/dev/null; }
trap cleanup EXIT

for ((i = 0; i < N; i++)); do
    gpu=${dg[$((i % ${#dg[@]}))]}
    ready="$OUT/draft_d$i.ready"; rm -f "$ready"
    CUDA_VISIBLE_DEVICES=$gpu DSD_DRAFT_LOG=${DSD_DRAFT_LOG:-0} "$PY" "$ROOT/scripts/run_draft_server.py" \
        --model "$DRAFT_MODEL" --drafter-id "d$i" --bind "tcp://127.0.0.1:0" \
        --gpu-frac "$frac" --dtype "$DRAFT_DTYPE" --max-model-len 2048 --ready-file "$ready" "${kind_args[@]}" \
        > "$OUT/draft_d$i.log" 2>&1 &
    pids+=($!)
    # Start drafters sharing a GPU one at a time: vLLM sizes its KV pool
    # from free memory at startup, so concurrent starts race each other.
    if (( (i + 1) % ${#dg[@]} == 0 )) || (( i + 1 == N )); then
        for ((j = 0; j <= i; j++)); do
            r="$OUT/draft_d$j.ready"
            for _ in $(seq 1 300); do
                [ -f "$r" ] && break
                kill -0 "${pids[$j]}" 2>/dev/null || { echo "drafter d$j died:"; tail -30 "$OUT/draft_d$j.log"; exit 1; }
                sleep 2
            done
            [ -f "$r" ] || { echo "drafter d$j not ready"; exit 1; }
        done
    fi
done
for ((i = 0; i < N; i++)); do addrs+=(--draft-addr "d$i=$(cat "$OUT/draft_d$i.ready")"); done
echo "drafters ready (N=$N, $per_gpu per GPU on $DGPUS, frac $frac)"
# Where everything ran: tier T1 = one drafter per GPU, T2 = several share one.
{
    printf '{"host": "%s", "target": {"gpu": %s, "model": "%s"}, "drafters_per_gpu": %s, ' \
        "$(uname -n)" "$TGPU" "$TARGET_MODEL" "$per_gpu"
    printf '"tier": "%s", "draft_kind": "%s", "draft_model": "%s", "drafters": {' \
        "$([ "$per_gpu" -gt 1 ] && echo T2 || echo T1)" "$DRAFT_KIND" "$DRAFT_MODEL"
    for ((i = 0; i < N; i++)); do
        printf '%s"d%d": {"gpu": %s, "addr": "%s"}' "$([ $i -gt 0 ] && echo ,)" "$i" \
            "${dg[$((i % ${#dg[@]}))]}" "$(cat "$OUT/draft_d$i.ready")"
    done
    printf '}, "gpus": "%s"}\n' "$(nvidia-smi --query-gpu=index,name --format=csv,noheader | tr '\n' ';')"
} > "$OUT/placement.json"

CUDA_VISIBLE_DEVICES=$TGPU "$PY" -m dsdreal.experiments.driver \
    --target-model "$TARGET_MODEL" "${addrs[@]}" --out "$OUT" "$@" 2>&1 | tee "$OUT/driver.log" \
    | grep -v "^\[rank0\]:  \|WARNING\|Loading safetensors"
rc=${PIPESTATUS[0]}
[ $rc -ne 0 ] && for p in "${pids[@]}"; do kill -USR1 "$p" 2>/dev/null; done && sleep 2
echo "driver exit $rc"
exit $rc
