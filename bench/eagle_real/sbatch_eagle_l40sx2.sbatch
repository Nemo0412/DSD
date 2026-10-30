#!/bin/bash
#SBATCH --job-name=eagle-l40sx2
#SBATCH --output=/home/ll5914/DSD-MLsys2027/logs/%x-%j.out
#SBATCH --error=/home/ll5914/DSD-MLsys2027/logs/%x-%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --account=torch_pr_111_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:2

# Qwen2-7B on GPU 0 (target), EAGLE head on GPU 1 (draft).
# Each round copies the hidden states across and sends the 60-node tree back.
# Then replay that trace in the simulator and print real vs sim.
set -euo pipefail
ROOT=/home/ll5914/DSD-MLsys2027
OUT=${ROOT}/results/eagle_l40sx2_${SLURM_JOB_ID}
mkdir -p "${OUT}" "${ROOT}/logs"
export EAGLE_ROOT=/scratch/ll5914/DSD-SIM/eagle/EAGLE
export PY=/scratch/yf3005/miniconda3/envs/maas-py310/bin/python
export PATH="$(dirname "${PY}"):${PATH}"
DEP=/scratch/ll5914/eagle-pydeps
if [ ! -d "${DEP}/accelerate" ]; then
  "${PY}" -m pip install --target "${DEP}" --no-deps 'accelerate==0.34.2'
fi
export PYTHONPATH="${DEP}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME=/scratch/ll5914/.huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "host=$(hostname)" | tee "${OUT}/host.txt"
nvidia-smi -L | tee -a "${OUT}/host.txt"

run_one() {
    local name=$1 prompts=$2 n=$3 maxtok=$4
    echo ">>> ${name} n=${n} max_new=${maxtok}"
    "${PY}" "${ROOT}/scripts/profile_eagle_tree_accept.py" \
        --base-model Qwen/Qwen2-7B-Instruct \
        --ea-model yuhuili/EAGLE-Qwen2-7B-Instruct \
        --prompts-file "${prompts}" \
        --output-jsonl "${OUT}/${name}.jsonl" \
        --id-prefix "${name}" \
        --max-new-tokens "${maxtok}" \
        --max-prompt-tokens 512 \
        --temperature 0.0 --top-p 1.0 \
        --depth 5 --total-token 60 --tree-top-k 10 \
        --dtype bfloat16 --limit "${n}" --seed 0 \
        --base-device 0 --draft-device 1
}

run_one gsm8k "${ROOT}/aws-trn/prompts/gsm8k_test.jsonl" 16 128
run_one cnndm "${ROOT}/aws-trn/prompts/cnndm_test.jsonl" 12 64
run_one humaneval "${ROOT}/aws-trn/prompts/humaneval_test.jsonl" 16 80

echo ">>> simulator replay"
unset PYTHONPATH
export DSPEC_SIM_ROOT="${ROOT}"
/scratch/ll5914/dsd-simvenv/bin/python "${ROOT}/scripts/compare_eagle_tree_sim.py" "${OUT}" \
  | tee "${OUT}/align.txt"
echo DONE
