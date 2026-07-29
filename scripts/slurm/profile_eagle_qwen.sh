#!/bin/bash
#SBATCH --job-name=eagle_qwen_smoke
#SBATCH --partition=l40s_public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# Optional site-specific account (do not hardcode usernames/accounts in-repo):
#   sbatch --account=YOUR_ACCOUNT scripts/slurm/profile_eagle_qwen.sh

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${REPO}/results/eagle_qwen_smoke"
PROMPTS="${REPO}/prompts/gsm8k_smoke_32.jsonl"
# Prefer an EAGLE-compatible Python (transformers~=4.53). Override with PY=...
PY="${PY:-python3}"
LOG_DIR="${LOG_DIR:-${REPO}/logs}"

# Required: path to SafeAILab/EAGLE checkout
if [[ -z "${EAGLE_ROOT:-}" ]]; then
  echo "ERROR: set EAGLE_ROOT to your local SafeAILab/EAGLE checkout" >&2
  exit 1
fi

if [[ -n "${HF_HOME:-}" ]]; then
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
  export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
fi
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export EAGLE_ROOT

mkdir -p "${OUT}" "${LOG_DIR}"

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
TRACE_PATH="${TRACE}" "${PY}" - <<'PY'
import json
import os
from pathlib import Path
p = Path(os.environ["TRACE_PATH"])
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
