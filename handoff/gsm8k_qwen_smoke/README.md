# GSM8K Qwen Smoke Logs (SpecDec++ + EAGLE)

Sanitized attachment package for simulator handoff. Contains **32** GSM8K prompts and hardware-profiled acceptance logs. No usernames, host paths, cluster accounts, tokens, or emails.

## Contents

| Path | Description |
|------|-------------|
| `prompts/gsm8k_smoke_32.jsonl` | 32 GSM8K smoke prompts |
| `specdecpp/gsm8k_smoke_hardware_acceptance.jsonl` | SpecDec++ / linear SD replay trace (`acceptance_seq`) |
| `specdecpp/gsm8k_smoke_details.jsonl` | Per-prompt iteration details from the profiler |
| `specdecpp/gsm8k_smoke_metrics.jsonl` | Aggregate acceptance metrics |
| `eagle/gsm8k_smoke_tree_accept.jsonl` | EAGLE tree SD replay trace (`tree_accept`) |

## Profiling setup (smoke substitute)

Paper defaults require gated Llama weights; this smoke run uses open Qwen models:

| Method | Draft / EA | Target / Base | Sampling |
|--------|------------|---------------|----------|
| SpecDec++ (linear) | `Qwen/Qwen2.5-1.5B` | `Qwen/Qwen2.5-7B` | temp=`0.0`, top_p=`1.0`, seed=`121`, `spec_tokens=5`, `max_new_tokens=128` |
| EAGLE (tree) | `yuhuili/EAGLE-Qwen2-7B-Instruct` | `Qwen/Qwen2-7B-Instruct` | temp=`0.0`, top_p=`1.0`, seed=`121`, depth=`5`, `max_new_tokens=128` |

## Schema reminders

- **SpecDec++:** `acceptance_seq` is a flat `0/1` list in draft-token order; accepted length per round is the longest leading run of ones.
- **EAGLE:** each `tree_accept[]` entry has `round`, `depth`, `candidates`, `verified`, `accepted_path_len`, `accepted_branch`, and timing fields.

## Reproduce (optional)

Repo scripts (paths are relative / env-driven; no personal defaults):

```bash
# SpecDec++
sbatch scripts/slurm/profile_specdecpp_qwen.sh

# EAGLE (requires SafeAILab/EAGLE checkout + transformers~=4.53)
export EAGLE_ROOT=/path/to/EAGLE
export PY=/path/to/python
sbatch scripts/slurm/profile_eagle_qwen.sh
```
