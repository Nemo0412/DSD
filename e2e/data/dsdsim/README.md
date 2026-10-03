Vendored from https://github.com/yfzzzyyls/dsdSim (MIT, see LICENSE) at the
commit in SOURCE_COMMIT, so real runs replay exactly the simulator's inputs:

- `gsm8k_trace_{1s,10s,master}.jsonl` -- from `traces/`: arrival_ms,
  prompt_tokens, target_tokens per request.
- `gsm8k_trace_source.jsonl` -- from `aws-trn/prompts/`: the GSM8K prompt
  text the traces were built from.
