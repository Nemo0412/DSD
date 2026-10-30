#!/bin/bash
#SBATCH --job-name=e2e-a100x4
#SBATCH --output=/home/ll5914/DSD-MLsys2027/logs/%x-%j.out
#SBATCH --error=/home/ll5914/DSD-MLsys2027/logs/%x-%j.err
#SBATCH --reservation=ll5914
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --partition=a100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --gres=gpu:a100:4
#SBATCH --time=10:00:00
#SBATCH hetjob
#SBATCH --reservation=ll5914
#SBATCH --account=torch_pr_111_general
#SBATCH --partition=l40s_public
#SBATCH --nodes=3
#SBATCH --ntasks=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:l40s:4
#SBATCH --time=10:00:00

# 4x A100 targets (ga001) + 12x L40S drafters (gl001-003).
# Same workload as the simulator rows: gamma=4, 0.5 req/s per drafter,
# GSM8K 16 / CNN-DM 12 / HumanEval 16, seed 0.
set -uo pipefail
ROOT=/home/ll5914/DSD-MLsys2027
OUT=${ROOT}/results/e2e_a100x4_l40sx12_${SLURM_JOB_ID}
mkdir -p "$OUT" "${ROOT}/logs"
export ROOT OUT
export DSD_PY=/scratch/yf3005/miniconda3/envs/vllm027/bin/python
export PATH="$(dirname "$DSD_PY"):$PATH"
export N_TARGETS=4 N_DRAFTERS=12 N_DRAFT_NODES=3
export TOPOLOGY=a100x4_l40sx12
export HF_HOME=/scratch/ll5914/.huggingface
bash "$ROOT/bench/e2e_table/run_campaign.sh"
