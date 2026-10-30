#!/bin/bash
# 2xA100 targets (GPU 0-1 on ga001) + 8xL40S drafts (gl001 and gl002).
# GPUs that this scale does not use stay on a small matmul.
set -uo pipefail

ROOT=${ROOT:-/home/ll5914/DSD-MLsys2027}
PY=${DSD_PY:-/scratch/yf3005/miniconda3/envs/vllm027/bin/python}
A100_JOB=${A100_JOB:-19648022}
DRAFT_JOB=${DRAFT_JOB:-19648021}
export A100_JOB DRAFT_JOB
OUT=${OUT:-$ROOT/results/e2e_a100x2_l40sx8_${A100_JOB}}
KEEP=${KEEP:-/scratch/ll5914/tmp/gpu_keepalive.py}
export TMPDIR=${TMPDIR:-/scratch/ll5914/tmp}
mkdir -p "$TMPDIR" "$OUT" "$ROOT/logs"

echo "10gpu starting 2xA100 + 8xL40S out=$OUT"

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

# Targets use A100 GPU 0 and 1. Keep GPU 2 and 3 busy.
srun --jobid="$A100_JOB" --overlap --ntasks=1 --cpu-bind=none --unbuffered \
  bash -c "for g in 2 3; do CUDA_VISIBLE_DEVICES=\$g $PY $KEEP >/dev/null 2>&1 & done; wait" \
  > "$OUT/keepalive_a100.log" 2>&1 &
keep_pids+=($!)

# Drafts use every GPU on gl001 and gl002. gl003 stays busy.
srun --jobid="$DRAFT_JOB" --overlap --cpu-bind=none --unbuffered \
  --ntasks=1 --ntasks-per-node=1 -w gl003 \
  bash -c "for g in 0 1 2 3; do CUDA_VISIBLE_DEVICES=\$g $PY $KEEP >/dev/null 2>&1 & done; wait" \
  > "$OUT/keepalive_l40s.log" 2>&1 &
keep_pids+=($!)
echo "keepalive sruns: ${keep_pids[*]}"

srun --jobid="$A100_JOB" --overlap --ntasks=1 --cpus-per-task=32 \
  --cpu-bind=none --unbuffered \
  env ROOT="$ROOT" OUT="$OUT" DSD_PY="$PY" \
      N_TARGETS=2 N_DRAFTERS=8 N_DRAFT_NODES=2 \
      DRAFTS_PER_NODE=4 DRAFT_NODELIST=gl[001-002] \
      TOPOLOGY=a100x2_l40sx8 DRAFT_JOB="$DRAFT_JOB" \
      HF_HOME=/scratch/ll5914/.huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      PATH="$(dirname "$PY"):$PATH" \
  bash "$ROOT/bench/e2e_table/run_campaign.sh"
rc=$?
echo "10gpu campaign rc=$rc"
exit "$rc"
