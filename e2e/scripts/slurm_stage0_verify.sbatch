#!/bin/bash
#SBATCH --job-name=dsd-stage0-verify
#SBATCH --account=torch_pr_111_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/yf3005/dsd-real/runs/stage0_slurm_%j.log

set -euo pipefail
cd /scratch/yf3005/dsd-real
source scripts/env.sh
export HF_HUB_OFFLINE=1
python scripts/verify_vllm_hooks.py
