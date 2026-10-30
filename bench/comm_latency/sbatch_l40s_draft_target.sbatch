#!/bin/bash
#SBATCH --job-name=dt-l40s
#SBATCH --account=torch_pr_111_general
#SBATCH --partition=l40s_public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=00:15:00
#SBATCH --output=/home/ll5914/DSD-MLsys2027/logs/%x-%j.out
#SBATCH --error=/home/ll5914/DSD-MLsys2027/logs/%x-%j.err
bash /home/ll5914/DSD-MLsys2027/bench/comm_latency/run_draft_target.sh l40s
