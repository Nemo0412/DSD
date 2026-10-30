#!/bin/bash
#SBATCH --job-name=dt-h100
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:15:00
#SBATCH --output=/home/ll5914/DSD-MLsys2027/logs/%x-%j.out
#SBATCH --error=/home/ll5914/DSD-MLsys2027/logs/%x-%j.err
bash /home/ll5914/DSD-MLsys2027/bench/comm_latency/run_draft_target.sh h100
