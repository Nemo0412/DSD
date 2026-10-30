#!/bin/bash
# After the stuck 16-GPU campaign exits: rerun EAGLE, then TinyLlama + Llama-2-7B
# on all 16 GPUs, then the 4-GPU campaign. Keep unused time off 0% utilization.
set -uo pipefail

ROOT=${ROOT:-/home/ll5914/DSD-MLsys2027}
PY=${DSD_PY:-/scratch/yf3005/miniconda3/envs/vllm027/bin/python}
A100_JOB=${A100_JOB:-19637436}
DRAFT_JOB=${DRAFT_JOB:-19637486}
export A100_JOB DRAFT_JOB
WAIT_PID=${WAIT_PID:-2503973}
KEEP=${KEEP:-/scratch/ll5914/tmp/gpu_keepalive.py}
OUT16=${OUT16:-$ROOT/results/e2e_a100x4_l40sx12_${A100_JOB}}
export TMPDIR=${TMPDIR:-/scratch/ll5914/tmp}
mkdir -p "$TMPDIR" "$ROOT/logs"

keep_pids=()
stop_keep() {
  for p in "${keep_pids[@]+"${keep_pids[@]}"}"; do
    kill "$p" 2>/dev/null || true
  done
  sleep 2
  for p in "${keep_pids[@]+"${keep_pids[@]}"}"; do
    kill -9 "$p" 2>/dev/null || true
  done
  keep_pids=()
}
start_keep() {
  stop_keep
  srun --jobid="$A100_JOB" --overlap --ntasks=1 --cpu-bind=none --unbuffered \
    bash -c "for g in 0 1 2 3; do CUDA_VISIBLE_DEVICES=\$g $PY $KEEP >/dev/null 2>&1 & done; wait" \
    > "$OUT16/keepalive_a100.log" 2>&1 &
  keep_pids+=($!)
  srun --jobid="$DRAFT_JOB" --overlap --cpu-bind=none --unbuffered \
    --ntasks=3 --ntasks-per-node=1 -w gl[001-003] \
    bash -c "for g in 0 1 2 3; do CUDA_VISIBLE_DEVICES=\$g $PY $KEEP >/dev/null 2>&1 & done; wait" \
    > "$OUT16/keepalive_l40s.log" 2>&1 &
  keep_pids+=($!)
  echo "keepalive ${keep_pids[*]}"
}
trap stop_keep EXIT

if [ "${SKIP_WAIT:-}" = 1 ]; then
  echo "followup starting now"
else
  echo "followup waiting for pid $WAIT_PID"
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 15
  done
  sleep 5
  echo "pid $WAIT_PID exited; 16-GPU eagle + tinyllama/llama2"
fi
start_keep

srun --jobid="$A100_JOB" --overlap --ntasks=1 --cpus-per-task=32 \
  --cpu-bind=none --unbuffered \
  env ROOT="$ROOT" OUT="$OUT16" DSD_PY="$PY" \
      N_TARGETS=4 N_DRAFTERS=12 N_DRAFT_NODES=3 \
      TOPOLOGY=a100x4_l40sx12 DRAFT_JOB="$DRAFT_JOB" \
      ONLY_PAIRS=eagle_qwen2_7b,tinyllama_llama2_7b \
      HF_HOME=/scratch/ll5914/.huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      PATH="$(dirname "$PY"):$PATH" \
  bash "$ROOT/bench/e2e_table/run_campaign.sh" \
  > "$ROOT/logs/e2e-a100x4-followup.out" 2>&1 &
camp=$!
phase=load
while kill -0 "$camp" 2>/dev/null; do
  eagle_go=0
  llama_go=0
  eagle_done=0
  [ -f "$OUT16/eagle_qwen2_7b/start.txt" ] && grep -q '>>> eagle_qwen2_7b gsm8k' "$OUT16/eagle_qwen2_7b/start.txt" && eagle_go=1
  [ -f "$OUT16/eagle_qwen2_7b/start.txt" ] && grep -q 'DONE eagle_qwen2_7b' "$OUT16/eagle_qwen2_7b/start.txt" && eagle_done=1
  [ -f "$OUT16/tinyllama_llama2_7b/start.txt" ] && grep -q '>>> tinyllama_llama2_7b gsm8k' "$OUT16/tinyllama_llama2_7b/start.txt" && llama_go=1
  if [ "$phase" = load ] && [ "$eagle_go" = 1 ]; then
    echo "eagle benchmark started; stop keepalive"
    stop_keep
    phase=eagle
  elif [ "$phase" = eagle ] && [ "$eagle_done" = 1 ] && [ "$llama_go" = 0 ]; then
    echo "between eagle and llama2; keepalive"
    start_keep
    phase=between
  elif [ "$phase" = between ] && [ "$llama_go" = 1 ]; then
    echo "llama2 benchmark started; stop keepalive"
    stop_keep
    phase=llama
  fi
  sleep 15
done
wait "$camp" || echo "16gpu followup rc=$?"
stop_keep

echo "starting 10-GPU campaign (2xA100 + 8xL40S)"
SKIP_WAIT=1 bash "$ROOT/bench/e2e_table/run_a100x2_l40sx8.sh"
echo "retry 16-GPU tinyllama/llama2 on drafts already up"
bash "$ROOT/bench/e2e_table/run_llama2_from_addrs.sh" || echo "llama2 retry rc=$?"
echo "starting 10-GPU campaign (2xA100 + 8xL40S)"
SKIP_WAIT=1 bash "$ROOT/bench/e2e_table/run_a100x2_l40sx8.sh"
echo "restart 10-GPU remaining pairs"
DRAFT_JOB=$(cat /scratch/ll5914/tmp/draft_job_id)
export DRAFT_JOB
A100_JOB=${A100_JOB:-19648022}
export A100_JOB
OUT10=$ROOT/results/e2e_a100x2_l40sx8_19648022
PY=${DSD_PY:-/scratch/yf3005/miniconda3/envs/vllm027/bin/python}
KEEP=${KEEP:-/scratch/ll5914/tmp/gpu_keepalive.py}
srun --jobid="$A100_JOB" --overlap --ntasks=1 --cpu-bind=none --unbuffered \
  bash -c "for g in 2 3; do CUDA_VISIBLE_DEVICES=\$g $PY $KEEP >/dev/null 2>&1 & done; wait" \
  > "$OUT10/keepalive_a100_restart.log" 2>&1 &
srun --jobid="$DRAFT_JOB" --overlap --cpu-bind=none --unbuffered \
  --ntasks=1 --ntasks-per-node=1 -w gl003 \
  bash -c "for g in 0 1 2 3; do CUDA_VISIBLE_DEVICES=\$g $PY $KEEP >/dev/null 2>&1 & done; wait" \
  > "$OUT10/keepalive_gl003_restart.log" 2>&1 &
srun --jobid="$A100_JOB" --overlap --ntasks=1 --cpus-per-task=32 \
  --cpu-bind=none --unbuffered \
  env ROOT="$ROOT" OUT="$OUT10" DSD_PY="$PY" \
      N_TARGETS=2 N_DRAFTERS=8 N_DRAFT_NODES=2 \
      DRAFTS_PER_NODE=4 DRAFT_NODELIST=gl[001-002] \
      TOPOLOGY=a100x2_l40sx8 DRAFT_JOB="$DRAFT_JOB" \
      ONLY_PAIRS=qwen7_14b,eagle_qwen2_7b,tinyllama_llama2_7b \
      HF_HOME=/scratch/ll5914/.huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      PATH="$(dirname "$PY"):$PATH" \
  bash "$ROOT/bench/e2e_table/run_campaign.sh"
echo "skip 4-GPU"
for h in ga001 gl001 gl002 gl003; do
  ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" "bash /scratch/ll5914/tmp/stop_keepalive.sh" || true
done
echo "starting 16-GPU routing x batching matrix"
INSIDE_ALLOC=1 bash "$ROOT/bench/e2e_table/run_policy_matrix.sh"
echo "followup done"
