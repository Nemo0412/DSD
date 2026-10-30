#!/bin/bash
# Queue a 1xA100 target + 3xL40S draft campaign behind the live 16-GPU run.
# Unused GPUs stay on a small matmul so the allocation is not idle.
# This process stays alive for the whole run (no exec) so a later waiter
# can block on this pid.
set -uo pipefail

ROOT=${ROOT:-/home/ll5914/DSD-MLsys2027}
PY=${DSD_PY:-/scratch/yf3005/miniconda3/envs/vllm027/bin/python}
A100_JOB=${A100_JOB:-19637436}
DRAFT_JOB=${DRAFT_JOB:-19637486}
WAIT_PID=${WAIT_PID:-2503973}
OUT=${OUT:-$ROOT/results/e2e_a100x1_l40sx3_${A100_JOB}}
KEEP=${KEEP:-/scratch/ll5914/tmp/gpu_keepalive.py}
export TMPDIR=${TMPDIR:-/scratch/ll5914/tmp}
mkdir -p "$TMPDIR" "$OUT" "$ROOT/logs"

if [ "${SKIP_WAIT:-}" = 1 ]; then
  echo "4gpu queue starting now"
else
  echo "4gpu queue waiting for pid $WAIT_PID"
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 30
  done
fi
# The launcher srun's child can exit a moment later.
sleep 5
echo "pid $WAIT_PID exited; starting 1xA100 + 3xL40S"

keep_pids=()
stop_keep() {
  for p in "${keep_pids[@]+"${keep_pids[@]}"}"; do
    kill "$p" 2>/dev/null || true
  done
  sleep 2
  for p in "${keep_pids[@]+"${keep_pids[@]}"}"; do
    kill -9 "$p" 2>/dev/null || true
  done
}
trap stop_keep EXIT

# A100 node: GPU 0 is the target. Keep GPUs 1-3 busy.
srun --jobid="$A100_JOB" --overlap --ntasks=1 --cpu-bind=none --unbuffered \
  bash -c "for g in 1 2 3; do CUDA_VISIBLE_DEVICES=\$g $PY $KEEP >/dev/null 2>&1 & done; wait" \
  > "$OUT/keepalive_a100.log" 2>&1 &
keep_pids+=($!)

# L40S: drafts use gl001 GPUs 0-2. One srun per node. A 3-task srun from
# inside the A100 step collapses onto one host and never starts.
srun --jobid="$DRAFT_JOB" --overlap --cpu-bind=none --unbuffered \
  --ntasks=1 --ntasks-per-node=1 -w gl001 \
  bash -c "CUDA_VISIBLE_DEVICES=3 $PY $KEEP >/dev/null 2>&1" \
  > "$OUT/keepalive_gl001.log" 2>&1 &
keep_pids+=($!)
for node in gl002 gl003; do
  srun --jobid="$DRAFT_JOB" --overlap --cpu-bind=none --unbuffered \
    --ntasks=1 --ntasks-per-node=1 -w "$node" \
    bash -c "for g in 0 1 2 3; do CUDA_VISIBLE_DEVICES=\$g $PY $KEEP >/dev/null 2>&1 & done; wait" \
    > "$OUT/keepalive_${node}.log" 2>&1 &
  keep_pids+=($!)
done
echo "keepalive sruns: ${keep_pids[*]}"

srun --jobid="$A100_JOB" --overlap --ntasks=1 --cpus-per-task=32 \
  --cpu-bind=none --unbuffered \
  env ROOT="$ROOT" OUT="$OUT" DSD_PY="$PY" \
      N_TARGETS=1 N_DRAFTERS=3 N_DRAFT_NODES=1 \
      DRAFTS_PER_NODE=3 DRAFT_NODELIST=gl001 \
      TOPOLOGY=a100x1_l40sx3 DRAFT_JOB="$DRAFT_JOB" \
      HF_HOME=/scratch/ll5914/.huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      PATH="$(dirname "$PY"):$PATH" \
  bash "$ROOT/bench/e2e_table/run_campaign.sh"
rc=$?
echo "4gpu campaign rc=$rc"
exit "$rc"
