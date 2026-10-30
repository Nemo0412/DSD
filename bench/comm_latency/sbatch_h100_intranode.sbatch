#!/bin/bash
#SBATCH --job-name=lat-h100
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:2
#SBATCH --time=00:25:00
#SBATCH --output=/home/ll5914/DSD-MLsys2027/logs/%x-%j.out
#SBATCH --error=/home/ll5914/DSD-MLsys2027/logs/%x-%j.err

bash /home/ll5914/DSD-MLsys2027/bench/comm_latency/run_intranode.sh h100
