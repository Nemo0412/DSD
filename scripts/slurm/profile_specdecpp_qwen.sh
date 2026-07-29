#!/bin/bash
#SBATCH --job-name=specdecpp_qwen_smoke
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=l40s_public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/ll5914/DSD-SIM/logs/specdecpp_qwen_%j.out
#SBATCH --error=/scratch/ll5914/DSD-SIM/logs/specdecpp_qwen_%j.err

set -euo pipefail

REPO=/home/ll5914/DSD-SIM
OUT=${REPO}/results/specdecpp_qwen_smoke
PROMPTS=${REPO}/prompts/gsm8k_smoke_32.jsonl
PY=/scratch/ll5914/conda_envs/SVD/bin/python

export HF_HOME=/scratch/ll5914/.huggingface
export TRANSFORMERS_CACHE=${HF_HOME}/hub
export HF_HUB_CACHE=${HF_HOME}/hub
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

mkdir -p "${OUT}" /scratch/ll5914/DSD-SIM/logs

echo "Host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi -L || true

# Paper default is Llama-2-chat 7B/70B (gated). Smoke uses open Qwen2.5 pair.
DRAFT=Qwen/Qwen2.5-1.5B
TARGET=Qwen/Qwen2.5-7B
SEED=121
SPEC_TOKENS=5
MAX_TOKENS=128
MAX_PROMPT_TOKENS=512

DETAILS=${OUT}/gsm8k_smoke_details.jsonl
METRICS=${OUT}/gsm8k_smoke_metrics.jsonl
TRACE=${OUT}/gsm8k_smoke_hardware_acceptance.jsonl

rm -f "${DETAILS}" "${METRICS}" "${TRACE}"

echo ">>> SpecDec++-style linear SD profiling (acceptance_seq)"
"${PY}" "${REPO}/src/acceptance/speculative_profiler.py" \
  --drafter-model "${DRAFT}" \
  --verifier-model "${TARGET}" \
  --prompts-file "${PROMPTS}" \
  --details-jsonl "${DETAILS}" \
  --metrics-jsonl "${METRICS}" \
  --spec-tokens "${SPEC_TOKENS}" \
  --max-tokens "${MAX_TOKENS}" \
  --max-prompt-tokens "${MAX_PROMPT_TOKENS}" \
  --temperature 0.0 \
  --top-p 1.0 \
  --seed "${SEED}" \
  --progress-bar

echo ">>> Export acceptance_seq traces"
"${PY}" "${REPO}/scripts/export_acceptance_seq_from_details.py" \
  --details "${DETAILS}" \
  --output "${TRACE}" \
  --arrival-stride-ms 25

# Stamp algorithm metadata onto each row
"${PY}" - <<'PY'
import json
from pathlib import Path
path = Path("/home/ll5914/DSD-SIM/results/specdecpp_qwen_smoke/gsm8k_smoke_hardware_acceptance.jsonl")
rows = []
for line in path.read_text().splitlines():
    if not line.strip():
        continue
    obj = json.loads(line)
    meta = dict(obj.get("metadata") or {})
    meta.update({
        "algorithm": "SpecDec++",
        "profiling_mode": "linear_speculative_decoding",
        "paper_model_pair": "meta-llama/Llama-2-7b-chat-hf + meta-llama/Llama-2-70b-chat-hf",
        "smoke_model_pair": "Qwen/Qwen2.5-1.5B + Qwen/Qwen2.5-7B",
        "drafter_model": "Qwen/Qwen2.5-1.5B",
        "verifier_model": "Qwen/Qwen2.5-7B",
        "dataset": "GSM8K",
        "dataset_split": "test_smoke_32",
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 121,
        "spec_tokens": 5,
        "max_new_tokens": 128,
        "acceptance_source": "hardware_profile",
        "note": "Llama gated; Qwen smoke substitute until HF access granted",
    })
    obj["metadata"] = meta
    rows.append(obj)
path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
print(f"wrote {len(rows)} rows -> {path}")
PY

echo ">>> Done. details=${DETAILS} trace=${TRACE}"
wc -l "${DETAILS}" "${TRACE}"
