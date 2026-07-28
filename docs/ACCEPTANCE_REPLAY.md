# Acceptance Sequence Replay

`DSD-Sim` prefers **hardware-collected `acceptance_seq` replay** for paper experiments.

1. **Replay mode (main path).**  
   Each JSONL request includes an `acceptance_seq` field: a list of `0/1` bits in draft-token order collected from real draft--target speculative decoding. When present, verification consumes this sequence and does **not** sample from a probabilistic model.

2. **Predictor fallback.**  
   If `acceptance_seq` is absent, the simulator uses the configured `.joblib` regressor/classifier or a fixed decaying acceptance rate (for sweeps only).

## Trace schema

```json
{
  "request_id": "gsm8k_00000",
  "arrival_ms": 0.0,
  "prompt_tokens": 36,
  "target_tokens": 114,
  "device_tier": "default",
  "acceptance_seq": [1, 1, 1, 0, 1, 1, 0],
  "metadata": {
    "acceptance_source": "hardware_profile",
    "drafter_model": "meta-llama/Llama-2-7b-hf",
    "verifier_model": "meta-llama/Llama-2-70b-hf"
  }
}
```

Semantics for a speculation window of size `gamma`:

- consume bits from `acceptance_seq`
- accept the longest leading run of ones (≤ `gamma`)
- if a zero appears inside the window, consume that reject bit and discard the remaining speculative tokens

## Hardware collection → export

```bash
# 1) Profile draft/target pair (writes details JSONL with accepted_flags)
python src/acceptance/speculative_profiler.py \
  --drafter-model meta-llama/Llama-2-7b-hf \
  --verifier-model meta-llama/Llama-2-70b-hf \
  --prompts-file prompts/gsm8k_test.jsonl \
  --details-jsonl results/run/gsm8k_test_details.jsonl \
  --metrics-jsonl results/run/gsm8k_test_metrics.jsonl \
  --spec-tokens 4

# 2) Export simulator traces with acceptance_seq
python scripts/export_acceptance_seq_from_details.py \
  --details results/run/gsm8k_test_details.jsonl \
  --output traces/gsm8k_hardware_acceptance.jsonl

# Or merge onto an existing arrival/workload file (row-aligned):
python scripts/export_acceptance_seq_from_details.py \
  --details results/run/gsm8k_test_details.jsonl \
  --workload traces/gsm8k_trace_1s.jsonl \
  --output traces/gsm8k_trace_1s_hardware_acceptance.jsonl
```

The end-to-end pipeline `src/acceptance/run_pipeline.sh` also exports `traces/*_hardware_acceptance.jsonl` after profiling.

Point your YAML workload path at a `*_hardware_acceptance.jsonl` file to enable replay mode.

## Sensitivity perturbations

```bash
# Rate bias (±10%/±20% ones)
python scripts/perturb_acceptance_seq.py \
  --input traces/gsm8k_trace_10s_with_acceptance.jsonl \
  --output /tmp/rate_m20.jsonl --mode rate-bias --delta -0.2 --report

# Burst shorten/extend
python scripts/perturb_acceptance_seq.py \
  --input traces/gsm8k_trace_10s_with_acceptance.jsonl \
  --output /tmp/burst_070.jsonl --mode burst --burst-factor 0.7 --report

# Cross-split transfer (donor sequences onto eval requests)
python scripts/perturb_acceptance_seq.py \
  --input traces/eval.jsonl --donor traces/donor.jsonl \
  --output /tmp/cross.jsonl --mode cross-split --report
```

Point the simulator YAML at the emitted JSONL to re-run end-to-end; `--report` also prints sequence stats and a first-order TPOT/throughput map from mean accepted tokens/window.
