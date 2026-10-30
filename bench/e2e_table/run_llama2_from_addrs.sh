#!/bin/bash
# Finish TinyLlama / Llama-2-7B on the 16-GPU allocation using draft
# servers that are already listening. Does not launch another draft srun.
set -uo pipefail
ROOT=${ROOT:-/home/ll5914/DSD-MLsys2027}
PY=${DSD_PY:-/scratch/yf3005/miniconda3/envs/vllm027/bin/python}
OUT=${OUT:-$ROOT/results/e2e_a100x4_l40sx12_19637436}
PAIR=$OUT/tinyllama_llama2_7b
TARGET=/scratch/ll5914/.huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9
TRACES=$ROOT/results/sim_a100x4_l40sx12/eagle_measured/traces
PROMPTS=$ROOT/aws-trn/prompts
export PYTHONPATH="/scratch/ll5914/py-packages:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=${HF_HOME:-/scratch/ll5914/.huggingface}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false VLLM_LOGGING_LEVEL=ERROR PYTHONUNBUFFERED=1
export PATH="/scratch/yf3005/miniconda3/envs/vllm027/bin:$PATH"
export TMPDIR=${TMPDIR:-/scratch/ll5914/tmp}
mkdir -p "$TMPDIR" "$PAIR"

target_pids=()
stop_targets() {
  for p in "${target_pids[@]+"${target_pids[@]}"}"; do kill "$p" 2>/dev/null || true; done
  sleep 3
  for p in "${target_pids[@]+"${target_pids[@]}"}"; do kill -9 "$p" 2>/dev/null || true; done
  target_pids=()
}
stop_drafts() {
  for h in gl001 gl002 gl003; do
    ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" \
      "bash /scratch/ll5914/tmp/stop_manual_drafts.sh" || true
  done
}
cleanup() {
  stop_targets
  stop_drafts
}
trap cleanup EXIT

n=$(ls "$OUT"/draft_d*.addr 2>/dev/null | wc -l)
echo "llama2 retry drafts=$n" | tee -a "$PAIR/start.txt"
if [ "$n" -lt 12 ]; then
  echo "FAIL llama2 retry missing draft addrs" | tee -a "$PAIR/start.txt"
  exit 1
fi

daddrs=()
for ((i = 0; i < 12; i++)); do
  daddrs+=(--draft-addr "d${i}=$(cat "$OUT/draft_d${i}.addr")")
done

taddrs=()
for ((j = 0; j < 4; j++)); do
  ready="$OUT/target_t${j}.ready"
  rm -f "$ready"
  CUDA_VISIBLE_DEVICES=$j "$PY" -m dsdreal.router.target_server \
    --target-model "$TARGET" --bind "tcp://127.0.0.1:0" --ready-file "$ready" --reuse \
    "${daddrs[@]}" --gamma 4 --rtt-ms 0 --dtype bfloat16 \
    --no-prefix-caching --gpu-frac 0.90 --max-model-len 1024 \
    --max-num-seqs 16 --warmup 1 --draft-timeout-s 180 \
    > "$PAIR/target_t${j}.log" 2>&1 &
  target_pids+=($!)
done

for ((j = 0; j < 4; j++)); do
  ready="$OUT/target_t${j}.ready"
  ok=0
  for _ in $(seq 1 450); do
    if [ -f "$ready" ]; then ok=1; break; fi
    if ! kill -0 "${target_pids[$j]}" 2>/dev/null; then break; fi
    sleep 2
  done
  if [ "$ok" != 1 ]; then
    echo "FAIL llama2 retry target t${j}" | tee -a "$PAIR/start.txt"
    tail -30 "$PAIR/target_t${j}.log" || true
    exit 1
  fi
  taddrs+=(--target-addr "t${j}=$(cat "$ready")")
  echo "target t${j} $(cat "$ready")" | tee -a "$PAIR/start.txt"
done

datasets=(gsm8k cnndm humaneval)
counts=(16 12 16)
prompt_files=(gsm8k_test.jsonl cnndm_test.jsonl humaneval_test.jsonl)
for k in 0 1 2; do
  ds=${datasets[$k]}
  ds_out="$PAIR/$ds"
  mkdir -p "$ds_out"
  echo ">>> tinyllama_llama2_7b $ds n=${counts[$k]}" | tee -a "$PAIR/start.txt"
  "$PY" -m dsdreal.router.frontend "${taddrs[@]}" --policy rr \
    --target-model "$TARGET" --n-drafters 12 --seed 0 \
    --trace "$TRACES/${ds}.jsonl" --prompts "$PROMPTS/${prompt_files[$k]}" \
    --n-requests "${counts[$k]}" --per-draft-rps 0.5 --out "$ds_out" \
    > "$ds_out/router.log" 2>&1 || echo "router rc $? for tinyllama_llama2_7b $ds" | tee -a "$PAIR/start.txt"
done
echo "DONE tinyllama_llama2_7b" | tee -a "$PAIR/start.txt"
rm -f "$PAIR/FAIL"
