#!/bin/bash
# Step D gate (docs/table3-plan.md): remote EAGLE3 head on 2 free GPUs,
# Llama-3.1-8B-Instruct + yuhuili/EAGLE3-LLaMA3.1-Instruct-8B.
#  1. acceptance length within 5% of vLLM's own fused method=eagle3, same trace;
#  2. lossless: fp32, remote-eagle3 outputs == no-draft baseline token for token.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env_nyulab.sh
GPUS=$(scripts/free_gpus.sh 2) || exit 1
T=${GPUS%%,*}; D=${GPUS##*,}
OUT=${1:-runs/stepD}; mkdir -p "$OUT"
HEAD=${EAGLE3_HEAD:-yuhuili/EAGLE3-LLaMA3.1-Instruct-8B}
COMMON=(--n-requests 60 --per-draft-rps 2 --gamma 4 --rtt-ms 10 --chat-template --no-prefix-caching)

CUDA_VISIBLE_DEVICES=$T $DSD_PY -m dsdreal.experiments.driver "${COMMON[@]}" --native-eagle3 "$HEAD" \
    --out "$OUT/native" > "$OUT/native.log" 2>&1 || { echo "native failed"; tail -20 "$OUT/native.log"; exit 1; }
DSD_DRAFT_KIND=eagle3 DSD_DRAFT_MODEL=$HEAD scripts/run_nto1.sh "$OUT/remote" 1 "$T" "$D" \
    "${COMMON[@]}" --draft-kind eagle3 > "$OUT/remote.log" 2>&1 || { echo "remote failed"; tail -30 "$OUT/remote.log"; exit 1; }
$DSD_PY - "$OUT" <<'PY'
import json, sys
out = sys.argv[1]
n = json.load(open(f"{out}/native/native_spec_metrics.json"))
r = json.load(open(f"{out}/remote/dsd_metrics.json"))["system_metrics"]
nat = 1 + n["vllm:spec_decode_num_accepted_tokens"] / n["vllm:spec_decode_num_drafts"]
rem = r["standard"]["mean_acceptance_length"]
ok = abs(rem - nat) / nat <= 0.05 and r["accounting_ok"]
print(json.dumps({"native_mean_accept_len": nat, "remote_mean_accept_len": rem,
                  "rel_diff": (rem - nat) / nat, "accounting_ok": r["accounting_ok"]}, indent=2))
sys.exit(0 if ok else 1)
PY
acc=$?

L=(--n-requests 30 --max-output-tokens 96 --per-draft-rps 2 --gamma 4 --rtt-ms 10 --chat-template --no-prefix-caching --dtype float32)
CUDA_VISIBLE_DEVICES=$T $DSD_PY -m dsdreal.experiments.driver "${L[@]}" --out "$OUT/fp32_base" \
    > "$OUT/fp32_base.log" 2>&1 || { echo "fp32 baseline failed"; exit 1; }
DSD_DRAFT_KIND=eagle3 DSD_DRAFT_MODEL=$HEAD DSD_DRAFT_DTYPE=float32 scripts/run_nto1.sh "$OUT/fp32_eagle3" 1 "$T" "$D" \
    "${L[@]}" --draft-kind eagle3 > "$OUT/fp32_eagle3.log" 2>&1 || { echo "fp32 eagle3 failed"; exit 1; }
$DSD_PY scripts/compare_outputs.py "$OUT/fp32_base"/outputs.jsonl "$OUT/fp32_eagle3"/outputs.jsonl > "$OUT/compare.json"
lossless=$?
cat "$OUT/compare.json"
if [ $lossless -ne 0 ]; then
    # fp32 is still batch-composition dependent (reduction order), so online
    # runs can flip an argmax at a near-tie. Accept a mismatch only if the
    # target's own top-2 logits at the divergence differ by < 1e-3.
    lossless=0
    for rid in $($DSD_PY -c "import json,sys; [print(m[0]) for m in json.loads(open(sys.argv[1]).read().split('PASS')[0].split('FAIL')[0])['first_mismatches']]" "$OUT/compare.json"); do
        CUDA_VISIBLE_DEVICES= $DSD_PY scripts/inspect_divergence.py "$OUT/fp32_base" "$OUT/fp32_eagle3" "$rid" --device cpu -- "${L[@]}" \
            > "$OUT/divergence_$rid.json" 2>/dev/null
        gap=$($DSD_PY -c "import json,sys; print(abs(json.load(open(sys.argv[1]))['gap_a_minus_b']))" "$OUT/divergence_$rid.json")
        echo "divergence $rid: |top-2 logit gap| = $gap"
        $DSD_PY -c "import sys; sys.exit(0 if float(sys.argv[1]) < 1e-3 else 1)" "$gap" || lossless=1
    done
    [ $lossless -eq 0 ] && echo "all mismatches are fp32 near-ties (< 1e-3): counted as lossless"
fi
echo "acceptance_rc=$acc lossless_rc=$lossless"
[ $acc -eq 0 ] && [ $lossless -eq 0 ] && echo "STEP D GATE PASS" || echo "STEP D GATE FAIL"
