#!/bin/bash
#SBATCH --job-name=prof-same
#SBATCH --output=/home/ll5914/DSD-MLsys2027/logs/%x-%j.out
#SBATCH --error=/home/ll5914/DSD-MLsys2027/logs/%x-%j.err
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:2

# Same node, two GPUs. Draft Qwen2.5-7B, target Qwen2.5-14B.
# Submit with -A, -p, --gres and --export=ALL,PROFILE_TAG=h100|l40s.
set -euo pipefail
ROOT=/home/ll5914/DSD-MLsys2027
TAG=${PROFILE_TAG:?}
OUT=${ROOT}/results/profile_${TAG}_${SLURM_JOB_ID}
mkdir -p "${OUT}"
export DSD_PY=/scratch/yf3005/miniconda3/envs/vllm027/bin/python
export PATH="$(dirname "$DSD_PY"):$PATH"
export DSD_DRAFT_MODEL=/scratch/ll5914/.huggingface/hub/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796
export DSD_TARGET_MODEL=Qwen/Qwen2.5-14B
export HF_HOME=/scratch/ll5914/.huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=${ROOT}
export VLLM_LOGGING_LEVEL=ERROR
echo "host=$(hostname) tag=${TAG}" | tee "${OUT}/host.txt"

cd "${ROOT}"
bash scripts/run_nto1.sh "${OUT}" 1 1 0 \
  --gamma 4 --rtt-ms 0 --n-requests 8 --warmup 1 \
  --dtype bfloat16 --no-prefix-caching --max-num-seqs 8 \
  > "${OUT}/run_nto1.log" 2>&1 &
RUN=$!
for _ in $(seq 1 900); do
  if [ -s "${OUT}/dsd_metrics.json" ]; then
    sleep 3
    kill "${RUN}" 2>/dev/null || true
    break
  fi
  kill -0 "${RUN}" 2>/dev/null || break
  sleep 2
done
wait "${RUN}" || true
"${DSD_PY}" bench/comm_latency/summarize_profile.py "${OUT}"
