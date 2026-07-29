#!/bin/bash
#SBATCH --job-name=eagle_qwen_smoke
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=l40s_public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/ll5914/DSD-SIM/logs/eagle_qwen_%j.out
#SBATCH --error=/scratch/ll5914/DSD-SIM/logs/eagle_qwen_%j.err

set -euo pipefail

REPO=/home/ll5914/DSD-SIM
EAGLE_ROOT=/scratch/ll5914/DSD-SIM/eagle/EAGLE
OUT=${REPO}/results/eagle_qwen_smoke
PROMPTS=${REPO}/prompts/gsm8k_smoke_32.jsonl
# Dedicated env: transformers 4.53.1 (EAGLE-compatible). Do NOT use SVD's transformers 5.x.
PY=/scratch/ll5914/DSD-SIM/envs/eagle/bin/python

export HF_HOME=/scratch/ll5914/.huggingface
export TRANSFORMERS_CACHE=${HF_HOME}/hub
export HF_HUB_CACHE=${HF_HOME}/hub
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export EAGLE_ROOT

mkdir -p "${OUT}" /scratch/ll5914/DSD-SIM/logs

echo "Host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi -L || true

# Official EAGLE supports Qwen2; paper often uses Vicuna (ungated) or LLaMA2-Chat (gated).
BASE=Qwen/Qwen2-7B-Instruct
EA=yuhuili/EAGLE-Qwen2-7B-Instruct
TRACE=${OUT}/gsm8k_smoke_tree_accept.jsonl

rm -f "${TRACE}"

echo ">>> EAGLE tree_accept profiling"
"${PY}" "${REPO}/scripts/profile_eagle_tree_accept.py" \
  --base-model "${BASE}" \
  --ea-model "${EA}" \
  --prompts-file "${PROMPTS}" \
  --output-jsonl "${TRACE}" \
  --max-new-tokens 128 \
  --max-prompt-tokens 512 \
  --temperature 0.0 \
  --top-p 1.0 \
  --depth 5 \
  --total-token 60 \
  --tree-top-k 10 \
  --seed 121 \
  --dtype float16

echo ">>> Done. trace=${TRACE}"
wc -l "${TRACE}"
"${PY}" - <<'PY'
import json
from pathlib import Path
p = Path("/home/ll5914/DSD-SIM/results/eagle_qwen_smoke/gsm8k_smoke_tree_accept.jsonl")
n = 0
rounds = 0
acc = 0.0
for line in p.read_text().splitlines():
    if not line.strip():
        continue
    o = json.loads(line)
    n += 1
    ta = o.get("tree_accept") or []
    rounds += len(ta)
    if ta:
        acc += sum(r.get("accepted_path_len", 0) for r in ta) / len(ta)
print(f"requests={n} total_rounds={rounds} mean_accept_path_len={acc/max(n,1):.3f}")
PY
