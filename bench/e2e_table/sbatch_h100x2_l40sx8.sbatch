#!/bin/bash
#SBATCH --job-name=e2e-h100x2
#SBATCH --output=/home/ll5914/DSD-MLsys2027/logs/%x-%j.out
#SBATCH --error=/home/ll5914/DSD-MLsys2027/logs/%x-%j.err
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h100:2
#SBATCH --time=08:00:00
#SBATCH hetjob
#SBATCH --reservation=ll5914
#SBATCH --account=torch_pr_111_general
#SBATCH --partition=l40s_public
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:l40s:4
#SBATCH --time=08:00:00

# 2x H100 targets + 8x L40S drafters. Submitted after the 16-GPU job so the
# two campaigns do not both claim gl001-002.
set -uo pipefail
ROOT=/home/ll5914/DSD-MLsys2027
OUT=${ROOT}/results/e2e_h100x2_l40sx8_${SLURM_JOB_ID}
mkdir -p "$OUT" "${ROOT}/logs"
export ROOT OUT
export DSD_PY=/scratch/yf3005/miniconda3/envs/vllm027/bin/python
export PATH="$(dirname "$DSD_PY"):$PATH"
export N_TARGETS=2 N_DRAFTERS=8 N_DRAFT_NODES=2
export TOPOLOGY=h100x2_l40sx8
export HF_HOME=/scratch/ll5914/.huggingface
bash "$ROOT/bench/e2e_table/run_campaign.sh"
