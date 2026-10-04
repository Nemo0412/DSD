#!/bin/bash
# Real Table 3 (from NYULAB) vs dsdSim, calibrated from the real runs.
#
#   scripts/final_compare.sh [LOCAL_OUT=runs/table3_final]
#
# 1. fetch runs/table3 from NYULAB (no logs/outputs);
# 2. calibrate the sim: target verify + prefill pooled over every vanilla/specpp
#    run; draft latency per method, both pooled and per N (N=4/7 put 2-4
#    drafter processes on one GPU -- tier T2 -- which the sim, one device per
#    drafter, cannot express; per-N draft calibration is the tier-aware variant);
# 3. sim sweeps (lockstep: our target step blocks on drafting) with the measured
#    per-position acceptance profile and round counts aligned (γ/L̄ output scale);
# 4. real_vs_sim_{pooled,tiered,replay}.md -- replay = tiered latency plus the
#    real run's own per-round acceptance (sim/make_config.py --replay-from).
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-runs/table3_final}
SCR=${SIM_ROOT:-$(pwd)/.sim}
SIMPY=${SIMPY:-$SCR/venv/bin/python}
export DSDSIM_ROOT=${DSDSIM_ROOT:-$SCR/dsdSim}
PY=${PY:-.venv/bin/python}
mkdir -p "$OUT"
[ "${FETCH:-1}" = 1 ] && ssh -o BatchMode=yes NYULAB "cd /mnt/hdd/external/dsd/dsd-real/runs/${REMOTE_RUN:-table3} && tar czf - --exclude='*.log' --exclude='*.ready' --exclude='outputs.jsonl' ." \
    2>/dev/null | tar xzf - -C "$OUT" || true

methods=(); for m in vanilla specpp eagle3; do ls "$OUT/${m}_N1/dsd_metrics.json" >/dev/null 2>&1 && methods+=("$m"); done
cal="$OUT/calibration.pooled.json"
runs=(); for m in vanilla specpp; do for n in 1 2 4 7; do [ -f "$OUT/${m}_N$n/steps.jsonl" ] && runs+=("$OUT/${m}_N$n"); done; done
$PY sim/calibrate_from_run.py "${runs[@]}" --method specpp -o "$cal" > "$OUT/calibration.pooled.fit.json"
$PY - "$cal" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1]))
c["draft"]["vanilla"].update({k: c["draft"]["specpp"][k] for k in ("base_ms", "ms_per_token", "ms_per_token_per_seq")})
json.dump(c, open(sys.argv[1], "w"), indent=2)
EOF
# EAGLE3 gets its own file: its target step also exports aux hidden states
# (a GPU->CPU copy per step), so its verify/prefill are fitted on its own runs,
# and its acceptance profile is its own (the head accepts differently from the
# 1B drafter; runs are gamma=4, positions past 4 are padded by the policy).
cal_e="$OUT/calibration.eagle.json"
if [[ " ${methods[*]} " == *" eagle3 "* ]]; then
    $PY sim/calibrate_from_run.py "$OUT"/eagle3_N{1,2,4,7} --method eagle --base "$cal" -o "$cal_e" > "$OUT/calibration.eagle.fit.json"
    $PY - "$OUT" <<'EOF'
import json, sys
from dsdreal.policy.window import AcceptanceProfile
out = sys.argv[1]
ctx, rows = {}, []
for n in (1, 2, 4, 7):
    for r in json.load(open(f"{out}/eagle3_N{n}/dsd_metrics.json"))["per_request"]:
        ctx[f"N{n}/" + r["request_id"]] = r["prompt_len"]
    for line in open(f"{out}/eagle3_N{n}/steps.jsonl"):
        if line.strip():
            d = json.loads(line); d["request_id"] = f"N{n}/" + d["request_id"]; rows.append(d)
AcceptanceProfile.from_steps(rows, ctx, edges=[64, 96]).to_json(f"{out}/eagle3_profile.json")
EOF
fi
calfor() { [ "$1" = eagle3 ] && echo "$cal_e" || echo "$cal"; }
proffor() { [ "$1" = eagle3 ] && echo "$OUT/eagle3_profile.json" || echo "$OUT/calib_g8/profile.json"; }

# The real prompts can be longer than the trace asked for (chat header +
# minimum body): give the sim the lengths that were actually run.
TRACE="$OUT/trace_real_lengths.jsonl"
$PY - "$OUT" "$TRACE" <<'EOF'
import json, sys
out, dst = sys.argv[1:]
real = {}
for m in ("vanilla", "specpp", "eagle3"):
    try:
        for r in json.load(open(f"{out}/{m}_N7/dsd_metrics.json"))["per_request"]:
            real.setdefault(r["request_id"], r["prompt_len"])
    except FileNotFoundError:
        pass
with open("data/dsdsim/gsm8k_trace_master.jsonl") as src, open(dst, "w") as f:
    for line in src:
        if line.strip():
            d = json.loads(line)
            d["prompt_tokens"] = real.get(d["request_id"], d["prompt_tokens"])
            f.write(json.dumps(d) + "\n")
EOF

scale() {  # method -> mean gamma / mean acceptance length (with bonus)
    $PY - "$OUT" "$1" <<'EOF'
import json, sys
out, m = sys.argv[1:]
L, g = [], []
for n in (1, 2, 4, 7):
    try:
        L.append(json.load(open(f"{out}/{m}_N{n}/dsd_metrics.json"))["system_metrics"]["standard"]["mean_acceptance_length"])
        g += [json.loads(l)["gamma"] for l in open(f"{out}/{m}_N{n}/steps.jsonl")]
    except FileNotFoundError:
        pass
print(round(sum(g) / len(g) / (sum(L) / len(L)), 3))
EOF
}
simname() { case $1 in eagle3) echo eagle3_chain ;; *) echo "$1" ;; esac; }
# same scenario on the sim side as the real runs used
scen=$($PY -c "import json,sys; print('--client-side' if json.load(open(sys.argv[1]))['config'].get('client_side') else '')" "$OUT/vanilla_N1/dsd_metrics.json")

for m in "${methods[@]}"; do
    sc=$(scale "$m")
    $SIMPY sim/run_sweep.py --methods "$(simname "$m")" --ns "1 2 4 7" --calibration "$(calfor "$m")" \
        --acceptance-profile "$(proffor "$m")" --mode lockstep --target-tokens-scale "$sc" \
        --out "$OUT/sim_pooled" --python "$SIMPY" --jobs 8 --force -- --trace "$TRACE" $scen > /dev/null
    for n in 1 2 4 7; do
        [ -f "$OUT/${m}_N$n/steps.jsonl" ] || continue
        kind=$m; [ "$m" = eagle3 ] && kind=eagle; [ "$m" = vanilla ] && kind=vanilla
        tc="$OUT/calibration.${m}_N$n.json"
        $PY sim/calibrate_from_run.py "$OUT/${m}_N$n" --method "$kind" --base "$(calfor "$m")" -o "$tc" > /dev/null
        # verify/prefill stay pooled: only the draft block is per tier
        $PY - "$(calfor "$m")" "$tc" <<'EOF'
import json, sys
pooled, tier = (json.load(open(p)) for p in sys.argv[1:])
tier["target"] = pooled["target"]
json.dump(tier, open(sys.argv[2], "w"), indent=2)
EOF
        $SIMPY sim/run_sweep.py --methods "$(simname "$m")" --ns "$n" --calibration "$tc" \
            --acceptance-profile "$(proffor "$m")" --mode lockstep --target-tokens-scale "$sc" \
            --out "$OUT/sim_tiered" --python "$SIMPY" --jobs 8 --force -- --trace "$TRACE" $scen > /dev/null
        # same, but acceptance replayed from this very run's token events:
        # identical gamma and per-round accepted counts, so what is left of
        # the error is the simulator's system model alone
        $SIMPY sim/run_sweep.py --methods "$(simname "$m")" --ns "$n" --calibration "$tc" \
            --mode lockstep --out "$OUT/sim_replay" --python "$SIMPY" --jobs 8 --force \
            -- --trace "$TRACE" $scen --replay-from "$OUT/${m}_N$n" > /dev/null
    done
done
$PY scripts/compare_real_sim.py "$OUT" "$OUT/sim_pooled" | tee "$OUT/real_vs_sim_pooled.md"
echo
$PY scripts/compare_real_sim.py "$OUT" "$OUT/sim_tiered" | tee "$OUT/real_vs_sim_tiered.md"
echo
$PY scripts/compare_real_sim.py "$OUT" "$OUT/sim_replay" | tee "$OUT/real_vs_sim_replay.md"
