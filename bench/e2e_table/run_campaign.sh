#!/bin/bash
# Real N-target / M-drafter speculative-decoding campaign.
# Targets run on this node (het group 0). Drafters are already the other
# het group: one node per 4 GPUs, launched by draft_node.sh.
#
# Required env: ROOT OUT DSD_PY N_TARGETS N_DRAFTERS
# Pairs are appended as records in $ROOT/bench/e2e_table/pairs.txt, or passed
# by the sbatch script via the PAIRS array below.
set -uo pipefail
ROOT=${ROOT:?}
OUT=${OUT:?}
PY=${DSD_PY:?}
N_TARGETS=${N_TARGETS:?}
N_DRAFTERS=${N_DRAFTERS:?}
N_DRAFT_NODES=${N_DRAFT_NODES:-$((N_DRAFTERS / 4))}
mkdir -p "$OUT" "$ROOT/logs"
export PYTHONPATH="/scratch/ll5914/py-packages:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=${HF_HOME:-/scratch/ll5914/.huggingface}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export EAGLE_ROOT=${EAGLE_ROOT:-/scratch/ll5914/DSD-SIM/eagle/EAGLE}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false
export VLLM_LOGGING_LEVEL=ERROR
export PYTHONUNBUFFERED=1

echo "host=$(hostname) targets=$N_TARGETS drafters=$N_DRAFTERS" | tee "$OUT/host.txt"
nvidia-smi -L | tee -a "$OUT/host.txt"

TRACES=$ROOT/results/sim_a100x4_l40sx12/eagle_measured/traces
PROMPTS=$ROOT/aws-trn/prompts

draft_pids=()
target_pids=()
stop_servers() {
  local pids=("${target_pids[@]}" "${draft_pids[@]}")
  for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done
  sleep 5
  for p in "${pids[@]}"; do kill -9 "$p" 2>/dev/null || true; done
  for p in "${pids[@]}"; do wait "$p" 2>/dev/null || true; done
  target_pids=()
  draft_pids=()
  sleep 8
}
trap stop_servers EXIT

wait_files() {  # count timeout_s glob
  local n=$1 t=$2
  local i=0
  while [ "$i" -lt "$t" ]; do
    local c
    c=$(compgen -G "$3" | wc -l)
    [ "$c" -ge "$n" ] && return 0
    sleep 2
    i=$((i + 2))
  done
  echo "timed out waiting for $n of $3 (have $c)"
  return 1
}

run_pair() {
  local name=$1 draft=$2 target=$3 kind=$4 gamma=$5 seqs=$6 frac=$7 chat=$8
  local pair_out="$OUT/$name"
  mkdir -p "$pair_out"
  if [ ! -d "$draft" ] || [ ! -d "$target" ]; then
    echo "SKIP $name missing weights draft=$draft target=$target" | tee "$pair_out/SKIP"
    return 0
  fi
  echo "===== $name kind=$kind gamma=$gamma =====" | tee "$pair_out/start.txt"
  stop_servers
  rm -f "$OUT"/draft_d*.ready "$OUT"/draft_d*.addr "$OUT"/target_t*.ready

  export DSD_DRAFT_MODEL="$draft" DSD_TARGET_MODEL="$target" DRAFT_KIND="$kind"
  export DRAFT_GPU_FRAC=0.85 MAX_MODEL_LEN=1024
  # DRAFT_JOB: drafts live in a separate allocation (the dummy hold), not a
  # hetjob group. Drop the outer step's SLURM_* so this srun fans out across
  # that job's nodes instead of collapsing onto one host.
  if [ -n "${DRAFT_JOB:-}" ]; then
    draft_place=()
    [ -n "${DRAFT_NODELIST:-}" ] && draft_place=(-w "$DRAFT_NODELIST")
    env -u SLURM_JOB_ID -u SLURM_JOB_NODELIST -u SLURM_NODELIST \
        -u SLURM_STEP_ID -u SLURM_STEP_NODELIST -u SLURM_NTASKS \
        -u SLURM_PROCID -u SLURM_NODEID -u SLURM_LOCALID \
        -u SLURM_TASKS_PER_NODE -u SLURM_STEP_GPUS -u SLURM_JOB_GPUS \
        -u SLURM_GPUS_ON_NODE -u SLURM_CLUSTER_NAME -u CUDA_VISIBLE_DEVICES \
        -u SLURM_NNODES -u SLURM_NPROCS -u SLURM_JOB_NUM_NODES \
        -u SLURM_JOBID -u SLURM_STEP_NUM_NODES -u SLURM_STEP_NUM_TASKS \
      srun --export=ALL --jobid="$DRAFT_JOB" --overlap --cpu-bind=none \
        --ntasks="$N_DRAFT_NODES" --ntasks-per-node=1 "${draft_place[@]}" \
        bash "$ROOT/bench/e2e_table/draft_node.sh" \
      > "$pair_out/draft_srun.log" 2>&1 &
  else
    srun --export=ALL --het-group=1 --ntasks="$N_DRAFT_NODES" --ntasks-per-node=1 \
      --cpus-per-task=8 bash "$ROOT/bench/e2e_table/draft_node.sh" \
      > "$pair_out/draft_srun.log" 2>&1 &
  fi
  draft_pids+=($!)

  if ! wait_files "$N_DRAFTERS" 1800 "$OUT/draft_d*.addr"; then
    echo "FAIL $name drafts" | tee "$pair_out/FAIL"
    tail -20 "$pair_out/draft_srun.log" || true
    for f in "$OUT"/draft_node*.host; do echo "-- $f"; cat "$f"; done
    stop_servers
    return 0
  fi

  daddrs=()
  for ((i = 0; i < N_DRAFTERS; i++)); do
    daddrs+=(--draft-addr "d${i}=$(cat "$OUT/draft_d${i}.addr")")
  done
  taddrs=()
  target_pids=()
  for ((j = 0; j < N_TARGETS; j++)); do
    ready="$OUT/target_t${j}.ready"
    rm -f "$ready"
    if [ "$kind" = eagle_tree ]; then
      CUDA_VISIBLE_DEVICES=$j "$PY" -m dsdreal.router.eagle_tree_server \
        --base-model "$target" --ea-model "$draft" \
        --bind "tcp://127.0.0.1:0" --ready-file "$ready" --reuse \
        "${daddrs[@]}" --dtype bfloat16 --device 0 \
        --total-token "${TREE_TOTAL_TOKEN:-60}" --depth "${TREE_DEPTH:-5}" \
        --top-k "${TREE_TOP_K:-10}" \
        > "$pair_out/target_t${j}.log" 2>&1 &
    else
      extra=()
      [ "$kind" = eagle3 ] && extra+=(--draft-kind eagle3)
      CUDA_VISIBLE_DEVICES=$j "$PY" -m dsdreal.router.target_server \
        --target-model "$target" --bind "tcp://127.0.0.1:0" --ready-file "$ready" --reuse \
        "${daddrs[@]}" --gamma "$gamma" --rtt-ms 0 --dtype bfloat16 \
        --no-prefix-caching --gpu-frac "$frac" --max-model-len 1024 \
        --max-num-seqs "$seqs" --warmup 1 --draft-timeout-s 180 \
        "${extra[@]}" > "$pair_out/target_t${j}.log" 2>&1 &
    fi
    target_pids+=($!)
    if ! wait_files 1 900 "$ready"; then
      echo "FAIL $name target t${j}" | tee "$pair_out/FAIL"
      tail -40 "$pair_out/target_t${j}.log" || true
      stop_servers
      return 0
    fi
    # ready file is tcp://127.0.0.1:port from the target process
    taddrs+=(--target-addr "t${j}=$(cat "$ready")")
    echo "target t${j} $(cat "$ready")" | tee -a "$pair_out/start.txt"
  done

  local -a datasets=(gsm8k cnndm humaneval)
  local -a counts=(16 12 16)
  local -a prompt_files=(gsm8k_test.jsonl cnndm_test.jsonl humaneval_test.jsonl)
  for k in 0 1 2; do
    local ds=${datasets[$k]} n=${counts[$k]}
    local ds_out="$pair_out/$ds"
    mkdir -p "$ds_out"
    echo ">>> $name $ds n=$n" | tee -a "$pair_out/start.txt"
    chat_args=()
    [ "$chat" = chat ] && chat_args+=(--chat-template)
    "$PY" -m dsdreal.router.frontend "${taddrs[@]}" --policy rr \
      --target-model "$target" --n-drafters "$N_DRAFTERS" --seed 0 \
      --trace "$TRACES/${ds}.jsonl" --prompts "$PROMPTS/${prompt_files[$k]}" \
      --n-requests "$n" --per-draft-rps 0.5 --out "$ds_out" \
      "${chat_args[@]}" > "$ds_out/router.log" 2>&1 || echo "router rc $? for $name $ds" | tee -a "$pair_out/start.txt"
  done
  echo "DONE $name" | tee -a "$pair_out/start.txt"
  stop_servers
}

# name draft target kind gamma max_num_seqs gpu_frac chat|plain
# ONLY_PAIRS, if set, is a comma-separated list. Empty runs every pair.
HF=$HF_HOME/hub
snap() { echo "$HF/models--${1}/snapshots/${2}"; }
want_pair() {
  [ -z "${ONLY_PAIRS:-}" ] && return 0
  case ",${ONLY_PAIRS}," in
    *",$1,"*) return 0 ;;
    *) echo "SKIP $1 (ONLY_PAIRS=$ONLY_PAIRS)"; return 1 ;;
  esac
}

if want_pair qwen15_7b; then
run_pair qwen15_7b \
  "$(snap Qwen--Qwen2.5-1.5B 8faed761d45a263340a0528343f099c05c9a4323)" \
  "$(snap Qwen--Qwen2.5-7B d149729398750b98c0af14eb82c78cfe92750796)" \
  vllm 4 16 0.90 plain
fi

if want_pair qwen7_14b; then
run_pair qwen7_14b \
  "$(snap Qwen--Qwen2.5-7B d149729398750b98c0af14eb82c78cfe92750796)" \
  "$(snap Qwen--Qwen2.5-14B 97e1e76335b7017d8f67c08a19d103c0504298c9)" \
  vllm 4 8 0.92 plain
fi

# Qwen2-7B-Instruct + EAGLE head. The head runs topK_genrate on the L40S
# nodes; tree_decoding stays on the A100/H100 targets.
if want_pair eagle_qwen2_7b; then
run_pair eagle_qwen2_7b \
  "$(snap yuhuili--EAGLE-Qwen2-7B-Instruct 1b9c01aa0c354e16e38ac6dd6b2b0b230be31e9f)" \
  "$(snap Qwen--Qwen2-7B-Instruct f2826a00ceef68f0f2b946d945ecc0477ce4450c)" \
  eagle_tree 4 1 0.90 chat
fi

# TinyLlama-1.1B and Llama-2-7B share one tokenizer.model. Llama-3.2-1B
# cannot draft for Llama-2 (vocab 128256 vs 32000). The raw Llama-2-7b
# repo is consolidated.pth; this loads the Hugging Face copy of the same weights.
any_snap() {
  local repo=$1
  local d="$HF/models--${repo}/snapshots"
  [ -d "$d" ] || return 1
  find "$d" -mindepth 1 -maxdepth 1 -type d | head -1
}
llama2_ready() {
  local s=$1 a b
  [ -n "$s" ] || return 1
  a="$s/model-00001-of-00002.safetensors"
  b="$s/model-00002-of-00002.safetensors"
  [ -f "$a" ] && [ -f "$b" ] || return 1
  [ "$(stat -Lc%s "$a")" -gt 9000000000 ] || return 1
  [ "$(stat -Lc%s "$b")" -gt 3000000000 ]
}
if want_pair tinyllama_llama2_7b; then
LLAMA_DRAFT=$(any_snap TinyLlama--TinyLlama-1.1B-Chat-v1.0 || true)
LLAMA_TARGET=""
for _ in $(seq 1 120); do
  LLAMA_TARGET=$(any_snap meta-llama--Llama-2-7b-hf || true)
  if [ -n "$LLAMA_DRAFT" ] && llama2_ready "$LLAMA_TARGET"; then
    break
  fi
  echo "waiting for Llama-2-7b-hf weights" | tee "$OUT/llama2_wait.txt"
  sleep 30
done
if [ -n "$LLAMA_DRAFT" ] && llama2_ready "$LLAMA_TARGET"; then
  run_pair tinyllama_llama2_7b "$LLAMA_DRAFT" "$LLAMA_TARGET" vllm 4 16 0.90 plain
else
  echo "SKIP tinyllama_llama2_7b Llama-2-7b-hf weights not ready" | tee "$OUT/llama_skip.txt"
fi
fi
echo "SKIP llama31_eagle3 Llama-3.1-8B stays gated; EAGLE3 head is not for Llama-2" | tee "$OUT/eagle3_skip.txt"

echo CAMPAIGN_DONE | tee "$OUT/CAMPAIGN_DONE"
"$PY" "$ROOT/bench/e2e_table/compare_table.py" --real "$OUT" --topology "$TOPOLOGY" \
  | tee "$OUT/compare.md"
