#!/bin/bash
# One L40S node: four draft servers, GPUs 0-3, ids d{base}..d{base+3}.
# SLURM_NODEID is 0..N-1 inside the draft het-group. Rewrites the bound
# 0.0.0.0 address to this node's 10.32.x IP so the A100/H100 targets can connect.
set -euo pipefail
ROOT=${ROOT:?}
OUT=${OUT:?}
PY=${DSD_PY:?}
KIND=${DRAFT_KIND:-vllm}
NLOCAL=${DRAFTS_PER_NODE:-4}
BASE=$(( ${SLURM_NODEID:-0} * NLOCAL ))
NODE_LOG="$OUT/draft_node${SLURM_NODEID:-0}"
mkdir -p "$OUT"
echo "draft node $(hostname) id=${SLURM_NODEID:-0} base=$BASE kind=$KIND" | tee "$NODE_LOG.host"

ip=$(ip -4 -o addr show | awk '{print $4}' | cut -d/ -f1 | grep '^10\.32\.' | head -1 || true)
if [ -z "$ip" ]; then
  ip=$(ip -4 -o addr show | awk '{print $4}' | cut -d/ -f1 | grep -v '^127\.' | head -1)
fi
echo "$ip" > "$NODE_LOG.ip"

kind_args=(--kind "$KIND")
if [ "$KIND" = eagle3 ] || [ "$KIND" = eagle_tree ]; then
  kind_args+=(--eagle-target-model "$DSD_TARGET_MODEL")
fi
if [ "$KIND" = eagle_tree ]; then
  kind_args+=(--tree-total-token "${TREE_TOTAL_TOKEN:-60}" \
              --tree-depth "${TREE_DEPTH:-5}" \
              --tree-top-k "${TREE_TOP_K:-10}")
fi

pids=()
cleanup() {
  for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done
  exit 0
}
trap cleanup TERM INT

for ((local = 0; local < NLOCAL; local++)); do
  i=$((BASE + local))
  ready="$OUT/draft_d${i}.ready"
  rm -f "$ready"
  # Drop a placeholder matmul before this GPU allocates the draft model.
  kp="/scratch/ll5914/tmp/keepbusy/$(hostname -s)_gpu${local}.pid"
  if [ -f "$kp" ]; then
    kill "$(cat "$kp")" 2>/dev/null || true
    rm -f "$kp"
    sleep 1
  fi
  CUDA_VISIBLE_DEVICES=$local "$PY" "$ROOT/scripts/run_draft_server.py" \
    --model "$DSD_DRAFT_MODEL" --drafter-id "d${i}" --bind "tcp://0.0.0.0:0" \
    --gpu-frac "${DRAFT_GPU_FRAC:-0.85}" --dtype bfloat16 \
    --max-model-len "${MAX_MODEL_LEN:-1024}" --ready-file "$ready" \
    "${kind_args[@]}" > "$OUT/draft_d${i}.log" 2>&1 &
  pids+=($!)
  for _ in $(seq 1 450); do
    [ -f "$ready" ] && break
    if ! kill -0 "${pids[-1]}" 2>/dev/null; then
      echo "drafter d${i} died" | tee -a "$NODE_LOG.host"
      tail -40 "$OUT/draft_d${i}.log" | tee -a "$NODE_LOG.host"
      exit 1
    fi
    sleep 2
  done
  [ -f "$ready" ] || { echo "drafter d${i} not ready"; exit 1; }
  raw=$(cat "$ready")
  port=${raw##*:}
  addr="tcp://${ip}:${port}"
  printf '%s\n' "$addr" > "$OUT/draft_d${i}.addr"
  printf 'd%s -> %s\n' "$i" "$addr" | tee -a "$NODE_LOG.host"
done

wait
