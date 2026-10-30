#!/bin/bash
#SBATCH --job-name=eagle-real
#SBATCH --output=/home/ll5914/DSD-MLsys2027/logs/%x-%j.out
#SBATCH --error=/home/ll5914/DSD-MLsys2027/logs/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# Real EAGLE tree (SafeAILab EaModel): Qwen2-7B-Instruct + EAGLE-Qwen2 head,
# depth 5, 60 tree nodes, top-k 10. Base and head share this GPU.
# Submit with -A -p --gres and --export=ALL,PROFILE_TAG=l40s|a100.
set -euo pipefail
ROOT=/home/ll5914/DSD-MLsys2027
TAG=${PROFILE_TAG:?}
OUT=${ROOT}/results/eagle_real_${TAG}_${SLURM_JOB_ID}
mkdir -p "${OUT}" "${ROOT}/logs"
export EAGLE_ROOT=/scratch/ll5914/DSD-SIM/eagle/EAGLE
# transformers 4.47: the vendored Qwen2 KV model overflows / mis-computes RoPE on transformers 5.
export PY=/scratch/yf3005/miniconda3/envs/maas-py310/bin/python
export PATH="$(dirname "${PY}"):${PATH}"
# accelerate is not in this env. Install it alone; a normal pip install
# upgrades torch/huggingface-hub and breaks transformers 4.47.
DEP=/scratch/ll5914/eagle-pydeps
rm -rf "${DEP}"
"${PY}" -m pip install --target "${DEP}" --no-deps 'accelerate==0.34.2'
export PYTHONPATH="${DEP}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME=/scratch/ll5914/.huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "host=$(hostname) tag=${TAG}" | tee "${OUT}/host.txt"
nvidia-smi -L | tee -a "${OUT}/host.txt"

BASE=Qwen/Qwen2-7B-Instruct
EA=yuhuili/EAGLE-Qwen2-7B-Instruct

run_one() {
    local name=$1 prompts=$2 n=$3 maxtok=$4
    echo ">>> ${name} n=${n} max_new=${maxtok}"
    "${PY}" "${ROOT}/scripts/profile_eagle_tree_accept.py" \
        --base-model "${BASE}" --ea-model "${EA}" \
        --prompts-file "${prompts}" \
        --output-jsonl "${OUT}/${name}.jsonl" \
        --max-new-tokens "${maxtok}" \
        --max-prompt-tokens 512 \
        --temperature 0.0 --top-p 1.0 \
        --depth 5 --total-token 60 --tree-top-k 10 \
        --dtype bfloat16 --limit "${n}" --seed 0
}

run_one gsm8k "${ROOT}/aws-trn/prompts/gsm8k_test.jsonl" 16 128
run_one cnndm "${ROOT}/aws-trn/prompts/cnndm_test.jsonl" 12 64
run_one humaneval "${ROOT}/aws-trn/prompts/humaneval_test.jsonl" 16 80
echo DONE
