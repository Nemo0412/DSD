#!/bin/bash
# Other routing x batching cells for the vLLM pairs.
# RR+FIFO is the campaign already running; this waits for that srun, then
# on the same held nodes runs:
#   fifo: random, jsq
#   lab:  rr, random, jsq
# EAGLE tree is not a vLLM scheduler, so it is not part of this matrix.
set -uo pipefail

ROOT=${ROOT:-/home/ll5914/DSD-MLsys2027}
PY=${DSD_PY:-/scratch/yf3005/miniconda3/envs/vllm027/bin/python}
export PATH="$(dirname "$PY"):$PATH"
export PYTHONPATH="/scratch/ll5914/py-packages:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=${HF_HOME:-/scratch/ll5914/.huggingface}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false
export VLLM_LOGGING_LEVEL=ERROR
export PYTHONUNBUFFERED=1
export TMPDIR=${TMPDIR:-/scratch/ll5914/tmp}
mkdir -p "$TMPDIR"

CAMPAIGN_PID=${CAMPAIGN_PID:-2363424}
A100_JOB=${A100_JOB:-19637436}
DRAFT_JOB=${DRAFT_JOB:-19637486}
N_TARGETS=${N_TARGETS:-4}
N_DRAFTERS=${N_DRAFTERS:-12}
N_DRAFT_NODES=${N_DRAFT_NODES:-3}
OUT=${OUT:-$ROOT/results/e2e_policy_${A100_JOB}}

if [ -z "${INSIDE_ALLOC:-}" ]; then
  echo "waiting for campaign pid $CAMPAIGN_PID"
  while kill -0 "$CAMPAIGN_PID" 2>/dev/null; do
    sleep 30
  done
  echo "campaign pid $CAMPAIGN_PID exited; starting policy matrix"
  squeue -u ll5914 -o '%.18i %.2t %.10M %R' || true
  exec srun --jobid="$A100_JOB" --overlap --ntasks=1 --cpus-per-task=32 \
    --cpu-bind=none --unbuffered \
    env INSIDE_ALLOC=1 ROOT="$ROOT" OUT="$OUT" DSD_PY="$PY" \
        DRAFT_JOB="$DRAFT_JOB" A100_JOB="$A100_JOB" \
        N_TARGETS="$N_TARGETS" N_DRAFTERS="$N_DRAFTERS" N_DRAFT_NODES="$N_DRAFT_NODES" \
        HF_HOME="$HF_HOME" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        PATH="$(dirname "$PY"):$PATH" \
    bash "$ROOT/bench/e2e_table/run_policy_matrix.sh"
fi

mkdir -p "$OUT" "$ROOT/logs"
echo "host=$(hostname) matrix out=$OUT" | tee "$OUT/host.txt"
nvidia-smi -L | tee -a "$OUT/host.txt"

TRACES=$ROOT/results/sim_a100x4_l40sx12/eagle_measured/traces
PROMPTS=$ROOT/aws-trn/prompts
HF=$HF_HOME/hub

draft_pids=()
target_pids=()
stop_targets() {
  for p in "${target_pids[@]+"${target_pids[@]}"}"; do kill "$p" 2>/dev/null || true; done
  sleep 5
  for p in "${target_pids[@]+"${target_pids[@]}"}"; do kill -9 "$p" 2>/dev/null || true; done
  for p in "${target_pids[@]+"${target_pids[@]}"}"; do wait "$p" 2>/dev/null || true; done
  target_pids=()
  sleep 5
}
stop_all() {
  stop_targets
  for p in "${draft_pids[@]+"${draft_pids[@]}"}"; do kill "$p" 2>/dev/null || true; done
  sleep 5
  for p in "${draft_pids[@]+"${draft_pids[@]}"}"; do kill -9 "$p" 2>/dev/null || true; done
  for p in "${draft_pids[@]+"${draft_pids[@]}"}"; do wait "$p" 2>/dev/null || true; done
  draft_pids=()
  sleep 5
}
trap stop_all EXIT

wait_files() {
  local n=$1 t=$2 i=0 c=0
  while [ "$i" -lt "$t" ]; do
    c=$(compgen -G "$3" | wc -l)
    [ "$c" -ge "$n" ] && return 0
    sleep 2
    i=$((i + 2))
  done
  echo "timed out waiting for $n of $3 (have $c)"
  return 1
}

start_drafts() {
  local pair_out=$1
  rm -f "$OUT"/draft_d*.ready "$OUT"/draft_d*.addr "$OUT"/draft_node*.host
  env -u SLURM_JOB_ID -u SLURM_JOB_NODELIST -u SLURM_NODELIST \
      -u SLURM_STEP_ID -u SLURM_STEP_NODELIST -u SLURM_NTASKS \
      -u SLURM_PROCID -u SLURM_NODEID -u SLURM_LOCALID \
      -u SLURM_TASKS_PER_NODE -u SLURM_STEP_GPUS -u SLURM_JOB_GPUS \
      -u SLURM_GPUS_ON_NODE -u SLURM_CLUSTER_NAME -u CUDA_VISIBLE_DEVICES \
      -u SLURM_NNODES -u SLURM_NPROCS -u SLURM_JOB_NUM_NODES \
      -u SLURM_JOBID -u SLURM_STEP_NUM_NODES -u SLURM_STEP_NUM_TASKS \
    srun --export=ALL --jobid="$DRAFT_JOB" --overlap --cpu-bind=none \
      --ntasks="$N_DRAFT_NODES" --ntasks-per-node=1 \
      bash "$ROOT/bench/e2e_table/draft_node.sh" \
    > "$pair_out/draft_srun.log" 2>&1 &
  draft_pids+=($!)
  wait_files "$N_DRAFTERS" 1800 "$OUT/draft_d*.addr"
}

start_targets() {
  local pair_out=$1 batching=$2 gamma=$3 seqs=$4 frac=$5
  local -a daddrs=() extra=()
  local j ready
  stop_targets
  rm -f "$OUT"/target_t*.ready
  for ((i = 0; i < N_DRAFTERS; i++)); do
    daddrs+=(--draft-addr "d${i}=$(cat "$OUT/draft_d${i}.addr")")
  done
  [ "$batching" = lab ] && extra+=(--batching lab)
  target_pids=()
  taddrs=()
  for ((j = 0; j < N_TARGETS; j++)); do
    ready="$OUT/target_t${j}.ready"
    rm -f "$ready"
    CUDA_VISIBLE_DEVICES=$j "$PY" -m dspec_real.router.target_server \
      --target-model "$DSD_TARGET_MODEL" --bind "tcp://127.0.0.1:0" --ready-file "$ready" --reuse \
      "${daddrs[@]}" --gamma "$gamma" --rtt-ms 0 --dtype bfloat16 \
      --no-prefix-caching --gpu-frac "$frac" --max-model-len 1024 \
      --max-num-seqs "$seqs" --warmup 1 --draft-timeout-s 180 \
      "${extra[@]}" > "$pair_out/target_${batching}_t${j}.log" 2>&1 &
    target_pids+=($!)
    if ! wait_files 1 900 "$ready"; then
      echo "FAIL targets batching=$batching t${j}" | tee "$pair_out/FAIL"
      tail -30 "$pair_out/target_${batching}_t${j}.log" || true
      return 1
    fi
    taddrs+=(--target-addr "t${j}=$(cat "$ready")")
    echo "target t${j} batching=$batching $(cat "$ready")" | tee -a "$pair_out/start.txt"
  done
}

run_frontend() {
  local pair_out=$1 policy=$2 batching=$3 chat=$4
  local cell="$pair_out/${policy}_${batching}"
  local -a datasets=(gsm8k cnndm humaneval)
  local -a counts=(16 12 16)
  local -a prompt_files=(gsm8k_test.jsonl cnndm_test.jsonl humaneval_test.jsonl)
  local k ds n
  echo "===== $(basename "$pair_out") policy=$policy batching=$batching =====" | tee -a "$pair_out/start.txt"
  for k in 0 1 2; do
    ds=${datasets[$k]}
    n=${counts[$k]}
    mkdir -p "$cell/$ds"
    chat_args=()
    [ "$chat" = chat ] && chat_args+=(--chat-template)
    echo ">>> $(basename "$pair_out") $policy $batching $ds n=$n" | tee -a "$pair_out/start.txt"
    "$PY" -m dspec_real.router.frontend "${taddrs[@]}" --policy "$policy" \
      --target-model "$DSD_TARGET_MODEL" --n-drafters "$N_DRAFTERS" --seed 0 \
      --trace "$TRACES/${ds}.jsonl" --prompts "$PROMPTS/${prompt_files[$k]}" \
      --n-requests "$n" --per-draft-rps 0.5 --out "$cell/$ds" \
      "${chat_args[@]}" > "$cell/$ds/router.log" 2>&1 \
      || echo "router rc $? for $policy $batching $ds" | tee -a "$pair_out/start.txt"
  done
}

run_vllm_pair() {
  local name=$1 draft=$2 target=$3 gamma=$4 seqs=$5 frac=$6 chat=$7
  local pair_out="$OUT/$name"
  mkdir -p "$pair_out"
  if [ ! -d "$draft" ] || [ ! -d "$target" ]; then
    echo "SKIP $name missing weights" | tee "$pair_out/SKIP"
    return 0
  fi
  echo "===== $name =====" | tee "$pair_out/start.txt"
  export DSD_DRAFT_MODEL="$draft" DSD_TARGET_MODEL="$target" DRAFT_KIND=vllm
  export DRAFT_GPU_FRAC=0.85 MAX_MODEL_LEN=1024
  stop_all
  trap stop_all EXIT
  if ! start_drafts "$pair_out"; then
    echo "FAIL $name drafts" | tee "$pair_out/FAIL"
    stop_all
    return 0
  fi
  # fifo targets serve the two policies the RR+FIFO campaign did not run
  if start_targets "$pair_out" fifo "$gamma" "$seqs" "$frac"; then
    run_frontend "$pair_out" random fifo "$chat"
    run_frontend "$pair_out" jsq fifo "$chat"
  fi
  if start_targets "$pair_out" lab "$gamma" "$seqs" "$frac"; then
    run_frontend "$pair_out" rr lab "$chat"
    run_frontend "$pair_out" random lab "$chat"
    run_frontend "$pair_out" jsq lab "$chat"
  fi
  echo "DONE $name" | tee -a "$pair_out/start.txt"
  stop_all
}

snap() { echo "$HF/models--${1}/snapshots/${2}"; }
any_snap() {
  local d="$HF/models--${1}/snapshots"
  [ -d "$d" ] || return 1
  find "$d" -mindepth 1 -maxdepth 1 -type d | head -1
}

run_vllm_pair qwen15_7b \
  "$(snap Qwen--Qwen2.5-1.5B 8faed761d45a263340a0528343f099c05c9a4323)" \
  "$(snap Qwen--Qwen2.5-7B d149729398750b98c0af14eb82c78cfe92750796)" \
  4 16 0.90 plain

run_vllm_pair qwen7_14b \
  "$(snap Qwen--Qwen2.5-7B d149729398750b98c0af14eb82c78cfe92750796)" \
  "$(snap Qwen--Qwen2.5-14B 97e1e76335b7017d8f67c08a19d103c0504298c9)" \
  4 8 0.92 plain

LLAMA_DRAFT=$(any_snap TinyLlama--TinyLlama-1.1B-Chat-v1.0 || true)
LLAMA_TARGET=$(any_snap meta-llama--Llama-2-7b-hf || true)
llama2_ready() {
  local s=$1 a b
  [ -n "$s" ] || return 1
  a="$s/model-00001-of-00002.safetensors"
  b="$s/model-00002-of-00002.safetensors"
  [ -f "$a" ] && [ -f "$b" ] || return 1
  [ "$(stat -Lc%s "$a")" -gt 9000000000 ] || return 1
  [ "$(stat -Lc%s "$b")" -gt 3000000000 ]
}
if [ -n "$LLAMA_DRAFT" ] && llama2_ready "$LLAMA_TARGET"; then
  run_vllm_pair tinyllama_llama2_7b "$LLAMA_DRAFT" "$LLAMA_TARGET" 4 16 0.90 plain
else
  echo "SKIP tinyllama_llama2_7b Llama-2-7b-hf weights missing" | tee "$OUT/llama_skip.txt"
fi

echo "SKIP eagle_qwen2_7b tree server is batch-1, not the vLLM scheduler" | tee "$OUT/tree_skip.txt"
echo MATRIX_DONE | tee "$OUT/MATRIX_DONE"
