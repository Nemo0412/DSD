#!/bin/bash
#SBATCH --job-name=dl-14b
#SBATCH --output=/home/ll5914/DSD-MLsys2027/logs/%x-%j.out
#SBATCH --error=/home/ll5914/DSD-MLsys2027/logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --account=torch_pr_111_general
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail
export HF_HOME=/scratch/ll5914/.huggingface
/scratch/yf3005/miniconda3/envs/vllm027/bin/python -c '
from huggingface_hub import snapshot_download
p = snapshot_download(
    "Qwen/Qwen2.5-14B",
    max_workers=2,
    allow_patterns=["*.json","*.safetensors","*.txt","merges.txt","vocab.json","tokenizer*"],
)
open("/home/ll5914/DSD-MLsys2027/logs/download_14b.done","w").write(p+"\n")
print("SNAPSHOT", p, flush=True)
'
