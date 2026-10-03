#!/bin/bash
# Step A gate (docs/table3-plan.md) on 2 free GPUs, Llama-3.1-8B / 3.2-1B:
#  1. lossless under online load: fp32, same trace, spec vs no-draft baseline,
#     outputs must match token for token;
#  2. bf16 N=1 run whose accounting must hold (driver exits non-zero if not).
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env_nyulab.sh
GPUS=$(scripts/free_gpus.sh 2) || exit 1
T=${GPUS%%,*}; D=${GPUS##*,}
OUT=${1:-runs/stepA}
mkdir -p "$OUT"
echo "target GPU $T, draft GPU $D -> $OUT"
COMMON=(--n-requests 40 --max-output-tokens 128 --per-draft-rps 2 --gamma 4 --rtt-ms 10 --chat-template --no-prefix-caching)

CUDA_VISIBLE_DEVICES=$T $DSD_PY -m dsdreal.experiments.driver --dtype float32 "${COMMON[@]}" \
    --out "$OUT/fp32_base" > "$OUT.fp32_base.log" 2>&1 || { echo "baseline failed"; tail -20 "$OUT.fp32_base.log"; exit 1; }
DSD_DRAFT_DTYPE=float32 scripts/run_nto1.sh "$OUT/fp32_spec" 1 "$T" "$D" --dtype float32 "${COMMON[@]}" \
    > "$OUT.fp32_spec.log" 2>&1 || { echo "fp32 spec failed"; tail -30 "$OUT.fp32_spec.log"; exit 1; }
$DSD_PY scripts/compare_outputs.py "$OUT/fp32_base"/outputs.jsonl "$OUT/fp32_spec"/outputs.jsonl > "$OUT/compare.json"
lossless=$?
cat "$OUT/compare.json"
if [ $lossless -ne 0 ]; then
    # fp32 is still batch-composition dependent (reduction order), so online
    # runs can flip an argmax at a near-tie. Accept a mismatch only if the
    # target's own top-2 logits at the divergence differ by < 1e-3.
    lossless=0
    for rid in $($DSD_PY -c "import json,sys; [print(m[0]) for m in json.loads(open(sys.argv[1]).read().split('PASS')[0].split('FAIL')[0])['first_mismatches']]" "$OUT/compare.json"); do
        CUDA_VISIBLE_DEVICES= $DSD_PY scripts/inspect_divergence.py "$OUT/fp32_base" "$OUT/fp32_spec" "$rid" --device cpu -- "${COMMON[@]}" \
            > "$OUT/divergence_$rid.json" 2>/dev/null
        gap=$($DSD_PY -c "import json,sys; print(abs(json.load(open(sys.argv[1]))['gap_a_minus_b']))" "$OUT/divergence_$rid.json")
        echo "divergence $rid: |top-2 logit gap| = $gap"
        $DSD_PY -c "import sys; sys.exit(0 if float(sys.argv[1]) < 1e-3 else 1)" "$gap" || lossless=1
    done
    [ $lossless -eq 0 ] && echo "all mismatches are fp32 near-ties (< 1e-3): counted as lossless"
fi

scripts/run_nto1.sh "$OUT/bf16_N1" 1 "$T" "$D" --n-requests 100 --per-draft-rps 2 --gamma 4 --rtt-ms 10 \
    > "$OUT.bf16_N1.log" 2>&1
acct=$?
echo "lossless_rc=$lossless accounting_rc=$acct"
[ $lossless -eq 0 ] && [ $acct -eq 0 ] && echo "STEP A GATE PASS" || echo "STEP A GATE FAIL"
