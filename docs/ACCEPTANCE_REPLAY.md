# Acceptance Sequence Replay

`DSD-Sim` supports two acceptance backends:

1. **Trace replay (preferred for paper experiments).**  
   Each JSONL request may include an `acceptance_seq` field: a list of `0/1` bits in draft-token order collected from hardware profiling (or synthesized for demos). When present, verification consumes this sequence and does **not** sample from a probabilistic model.

2. **Predictor fallback.**  
   If `acceptance_seq` is absent, the simulator uses the configured `.joblib` regressor/classifier or a fixed decaying acceptance rate.

## Trace schema

```json
{
  "request_id": "gsm8k_00000",
  "arrival_ms": 0.0,
  "prompt_tokens": 36,
  "target_tokens": 114,
  "device_tier": "default",
  "acceptance_seq": [1, 1, 1, 0, 1, 1, 0],
  "metadata": {"client_id": "client_000"}
}
```

Semantics for a speculation window of size `gamma`:

- consume bits from `acceptance_seq`
- accept the longest leading run of ones (≤ `gamma`)
- if a zero appears inside the window, consume that reject bit and discard the remaining speculative tokens

## Generate / attach sequences

Hardware path (preferred):

```bash
# profile draft/target pair, then map logs into acceptance_seq per request
python profile_acceptance_detailed.py ...
```

Demo / schema path (no GPU required):

```bash
python scripts/augment_trace_with_acceptance.py \
  --input traces/gsm8k_trace_1s.jsonl \
  --output traces/gsm8k_trace_1s_with_acceptance.jsonl \
  --gamma 4 --mean-accept 3.0 --seed 7
```

Point your YAML workload path at the `*_with_acceptance.jsonl` file to enable replay mode.
