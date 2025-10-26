#!/usr/bin/env bash
set -euo pipefail
SCENARIOS=(
  "gsm8k:experiments/results/dynamic_gamma_mix/configs/specpp_drafts400_seed123.yaml:/home/external/dsdSim/traces/gsm8k_trace_200prompts.jsonl"
  "cnndm:experiments/results/dynamic_gamma_mix/configs/specpp_drafts400_seed123.yaml:/home/external/dsdSim/traces/cnndm_trace_200prompts.jsonl"
  "humaneval:experiments/results/dynamic_gamma_mix/configs/specpp_drafts400_seed123.yaml:/home/external/dsdSim/traces/humaneval_trace_100prompts.jsonl"
)
LOG_DIR=data/gamma_oracle/logs
mkdir -p "$LOG_DIR"

for entry in "${SCENARIOS[@]}"; do
  IFS=':' read -r SCENARIO CONFIG TRACE <<<"$entry"
  TMP_CFG=$(mktemp)
  python - <<PY
import yaml, pathlib
cfg_path = pathlib.Path("$CONFIG")
data = yaml.safe_load(cfg_path.read_text())
data["trace_path"] = "$TRACE"
yaml.safe_dump(data, open("$TMP_CFG", "w"), sort_keys=False)
PY
  LOG_PATH="$LOG_DIR/${SCENARIO}_$(date +%Y%m%d_%H%M%S).log"
  echo "Running $SCENARIO, logs -> $LOG_PATH"
  python -m src.experiments.gamma_oracle_runner \
    --config "$TMP_CFG" \
    --scenario-id "$SCENARIO" \
    --drafter-counts 800,1000,1200,1400 \
    --rtt-multipliers 1.0,1.2,1.5,2.0,5.0,8.0,10.0,15.0 \
    --gammas 1,2,3,4,5,6,7,8 \
    --max-conversations 300 \
    --seed 123 \
    --output-dir data/gamma_oracle > "$LOG_PATH" 2>&1
  rm -f "$TMP_CFG"
done
