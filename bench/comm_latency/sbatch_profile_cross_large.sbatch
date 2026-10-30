#!/bin/bash
#SBATCH --job-name=prof-xbig
#SBATCH --output=/home/ll5914/DSD-MLsys2027/logs/%x-%j.out
#SBATCH --error=/home/ll5914/DSD-MLsys2027/logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:h100:1
#SBATCH hetjob
#SBATCH --account=torch_pr_111_general
#SBATCH --partition=l40s_public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=01:00:00

# Target Qwen2.5-14B on H100, draft Qwen2.5-7B on L40S. No injected sleep.
set -euo pipefail
ROOT=/home/ll5914/DSD-MLsys2027
OUT=${ROOT}/results/profile_xbig_${SLURM_JOB_ID}
mkdir -p "${OUT}"
export ROOT OUT
export DSD_PY=/scratch/yf3005/miniconda3/envs/vllm027/bin/python
export PATH="$(dirname "$DSD_PY"):$PATH"
export DSD_DRAFT_MODEL=${DSD_DRAFT_MODEL:-/scratch/ll5914/.huggingface/hub/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796}
export DSD_TARGET_MODEL=${DSD_TARGET_MODEL:-Qwen/Qwen2.5-14B}
export HF_HOME=/scratch/ll5914/.huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=${ROOT}
export VLLM_LOGGING_LEVEL=ERROR

srun --export=ALL --het-group=1 --ntasks=1 --cpus-per-task=8 bash -c '
set -euo pipefail
cd "$ROOT"
ready="$OUT/draft.ready"
rm -f "$ready"
echo "draft_host=$(hostname)" > "$OUT/draft_host.txt"
CUDA_VISIBLE_DEVICES=0 "$DSD_PY" scripts/run_draft_server.py \
  --model "$DSD_DRAFT_MODEL" --drafter-id d0 --bind tcp://0.0.0.0:0 \
  --gpu-frac 0.85 --dtype bfloat16 --max-model-len 2048 --ready-file "$ready" \
  > "$OUT/draft.log" 2>&1 &
echo $! > "$OUT/draft.pid"
for _ in $(seq 1 450); do
  [ -f "$ready" ] && break
  kill -0 $(cat "$OUT/draft.pid") 2>/dev/null || { echo draft died; tail -40 "$OUT/draft.log"; exit 1; }
  sleep 2
done
raw=$(cat "$ready")
port=${raw##*:}
ip=$(ip -4 -o addr show | awk "{print \$4}" | cut -d/ -f1 | grep "^10\\.32\\." | head -1)
[ -n "$ip" ] || ip=$(ip -4 -o addr show | awk "{print \$4}" | cut -d/ -f1 | grep -v "^127\." | head -1)
echo "tcp://${ip}:${port}" > "$OUT/draft.addr"
echo "draft addr $(cat "$OUT/draft.addr") on $(hostname)"
wait $(cat "$OUT/draft.pid")
' > "${OUT}/draft_wrap.log" 2>&1 &
DRAFT_WRAP=$!

for _ in $(seq 1 450); do
  [ -f "${OUT}/draft.addr" ] && break
  sleep 2
done
[ -f "${OUT}/draft.addr" ] || { echo "draft address missing"; cat "${OUT}/draft_wrap.log"; exit 1; }
ADDR=$(cat "${OUT}/draft.addr")
echo "target_host=$(hostname)" | tee "${OUT}/target_host.txt"
echo "target $(hostname) -> ${ADDR}"

srun --export=ALL --het-group=0 --ntasks=1 --cpus-per-task=8 \
  env CUDA_VISIBLE_DEVICES=0 "${DSD_PY}" -m dsdreal.experiments.driver \
  --target-model "${DSD_TARGET_MODEL}" --draft-addr "d0=${ADDR}" \
  --gamma 4 --rtt-ms 0 --n-requests 8 --warmup 1 \
  --dtype bfloat16 --no-prefix-caching --max-num-seqs 8 \
  --out "${OUT}" > "${OUT}/driver.log" 2>&1 &
DRIVER=$!

# Metrics land before the process exits. vLLM can hang on NCCL shutdown,
# so stop the driver once the run file exists and release both nodes.
for _ in $(seq 1 900); do
  if [ -s "${OUT}/dsd_metrics.json" ]; then
    sleep 3
    kill "${DRIVER}" 2>/dev/null || true
    break
  fi
  kill -0 "${DRIVER}" 2>/dev/null || break
  sleep 2
done
wait "${DRIVER}" || true
kill "${DRAFT_WRAP}" 2>/dev/null || true
sleep 2
wait "${DRAFT_WRAP}" || true
"${DSD_PY}" "${ROOT}/bench/comm_latency/summarize_profile.py" "${OUT}"
