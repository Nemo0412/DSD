#!/bin/bash
#SBATCH --job-name=prof-same
#SBATCH --output=/home/ll5914/DSD-MLsys2027/logs/%x-%j.out
#SBATCH --error=/home/ll5914/DSD-MLsys2027/logs/%x-%j.err
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2

# Submit twice, once per partition. The wrapper sets ACCOUNT/PARTITION/TAG.
set -euo pipefail
ROOT=/home/ll5914/DSD-MLsys2027
TAG=${PROFILE_TAG:?}
OUT=${ROOT}/results/profile_${TAG}_${SLURM_JOB_ID}
export DSD_PY=/scratch/yf3005/miniconda3/envs/vllm027/bin/python
export PATH="$(dirname "$DSD_PY"):$PATH"
export DSD_DRAFT_MODEL=/scratch/ll5914/.huggingface/hub/models--Qwen--Qwen2.5-1.5B/snapshots/8faed761d45a263340a0528343f099c05c9a4323
export DSD_TARGET_MODEL=/scratch/ll5914/.huggingface/hub/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796
export HF_HOME=/scratch/ll5914/.huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=${ROOT}
export VLLM_LOGGING_LEVEL=ERROR

cd "${ROOT}"
bash scripts/run_nto1.sh "${OUT}" 1 1 0 \
  --gamma 4 --rtt-ms 0 --n-requests 8 --warmup 1 \
  --dtype bfloat16 --no-prefix-caching --max-num-seqs 8
"${DSD_PY}" bench/comm_latency/summarize_profile.py "${OUT}"
