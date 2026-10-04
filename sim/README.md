# sim/: dsdSim configured to match the dsd-real setup

This directory runs the unmodified DSD-Sim simulator
([yfzzzyyls/dsdSim](https://github.com/yfzzzyyls/dsdSim), MIT) on our small
"Table 3" setup and writes results with the same metric names as
`scripts/make_table3.py`'s **dsdSim convention** block. The clone itself is
never edited. Everything specific to us is in this directory: a config
generator, a thin runtime shim, and a latency calibration file.

**The latency numbers in `calibration.json` are placeholders.** The
results under `results/` show that the pipeline works. They are not a
prediction yet. See [Calibrating](#calibrating).

| file | what |
|---|---|
| `calibration.json` | every latency constant the sim uses (PLACEHOLDERS) |
| `make_config.py` | (method, N, calibration, per_draft_rps, rtt, n_requests, seed, mode) → `config.yaml` + retimed `trace.jsonl` |
| `dsdsim_shim.py` | import fixes + calibrated latency provider + monkeypatches (no clone edits) |
| `run_sim.py` | runs one config through the shim → result JSON (table row, diagnostics, per-request records) |
| `run_sweep.py` | methods × N sweep → `results/<method>_N<n>.json` + `results/table3_sim.md` |
| `results/` | placeholder-calibrated async sweep; `results/lockstep/` = the same sweep in lockstep mode |

## Setup

```bash
SCRATCH=/tmp/claude-1000/-home-scatyf3-dsd-real/296ff36d-0e9c-405f-8fa4-47eb7d19d1d3/scratchpad   # any dir
git clone --depth 1 https://github.com/yfzzzyyls/dsdSim $SCRATCH/dsdSim
uv venv --python 3.11 $SCRATCH/simvenv            # sklearn 1.5.2 has no wheels for 3.13+/3.14
uv pip install --python $SCRATCH/simvenv/bin/python simpy "numpy<2.1" scikit-learn==1.5.2 joblib==1.5.2 pyyaml
export DSDSIM_ROOT=$SCRATCH/dsdSim
SIMPY=$SCRATCH/simvenv/bin/python
```

The venv only needs these packages (no torch or vllm). `scikit-learn==1.5.2`
is the version the acceptance `.joblib` was pickled with (it matches the
clone's `requirements.txt`). Tested with simpy 4.1.2, numpy 2.0.2, and
Python 3.11.14.

## Running

```bash
# full sweep: vanilla specpp eagle3 eagle3_chain x N=1 2 4 7  (~15 s wall with --jobs 8)
$SIMPY sim/run_sweep.py --jobs 8
$SIMPY sim/run_sweep.py --jobs 8 --mode lockstep --out sim/results/lockstep
# one config by hand
$SIMPY sim/make_config.py --method specpp --n-drafters 4 --out-dir /tmp/specpp_N4
$SIMPY sim/run_sim.py /tmp/specpp_N4/config.yaml -o /tmp/specpp_N4.json --log /tmp/specpp_N4/sim.log
# re-render the table only
$SIMPY sim/run_sweep.py --table-only
```

`run_sweep.py` options: `--calibration`, `--per-draft-rps` (1.0), `--rtt-ms`
(10), `--n-requests` (default `min(400, 120*N)`), `--seed` (0),
`--mode async|lockstep`, `--acceptance-profile`, `--target-tokens-scale`,
`--force` (re-run existing results), `--python` (sim venv interpreter), and
extra `make_config.py` arguments after `--`.

**How dsdSim is normally invoked:** `python src/sim.py --config X.yaml`
(`sim.py:main`, which prints a `===METRICS_JSON===` block). Without VIDUR
that command **cannot even start**. `src/performance/__init__.py` imports
`vidur_realtime.py`, which raises `FileNotFoundError` at import time when
`thirdparty/Sai_speculative_vidur` is missing. `run_sim.py` loads the
same `sim` module through the shim and calls the same functions `main` does
(`load_config_from_mapping` → `run` → `Metrics.summary` →
`_collect_metrics_json`).

## Mapping: our real setup ↔ sim config

| real (dsd-real) | sim config written by `make_config.py` |
|---|---|
| 1 target, Llama-3.1-8B bf16, A6000 | one device `t0`, `role: target`. Latency comes from `dsdreal_calib.target` (calibration.json) |
| vLLM max_num_seqs 64 | `t0.batch_size: 64`, `scheduler.{prefill,decode}.max_batch_requests: 64` |
| continuous batching (never waits) | `t0.batch_window_ms: 0.01`, `scheduler.*.max_wait_ms: 0`. A window of 0 would mean one job per batch (`sim.py:1743-1751`), so it is kept just above 0 |
| N drafters (Llama-3.2-1B / EAGLE3 head), RR routing | one **virtual** draft device per request, `d{p}_r{i}`, where `p = i mod N` is the physical drafter (see *Arrival process*) |
| RTT 10 ms, no jitter (netem sleep) | per virtual drafter link `forward_latency_ms = response_latency_ms = 5`, `network.bandwidth_mbps: 0` (no serialization term), `link_capacity: 1e6` (no link queueing), `jitter_pct: 0` |
| trace `gsm8k_trace_master.jsonl`, first `min(400,120N)` rows | `trace_path` → retimed copy with the same `prompt_tokens`, `target_tokens`, and `request_id` |
| open-loop Poisson at `per_draft_rps*N`, seed | `arrival_ms` from the **same generator** as `dsdreal/workload/trace.py::poisson_arrivals` (identical arrival times for the same seed); `think_time: constant 0 ms` |
| Vanilla, γ=4 | `gamma: 4`, `speculation.gamma_policy.type: constant` |
| SpecDec++ (min 2, max 8, threshold 0.7) | `gamma_policy: {type: specpp, min_gamma: 2, max_gamma: 8, stop_threshold: 0.7}` |
| EAGLE3 (remote head, chain depth 4) | `speculation.framework: eagle`, `eagle: {beam_width: 1, max_depth: 4, min_depth: 1, prune 0, load_control off}` (**broken for our purpose**, see below). Also `eagle3_chain` = vanilla framework, γ=4, EAGLE3 draft latency |
| acceptance | `speculation.acceptance.model: src/acceptance/llama3_1b_vs_8b_500.joblib`, classifier on, `default_rate 0.7` |

## Latency code paths and calibration

With VIDUR absent, dsdSim gets every latency from a *performance provider*.
There are three paths:

| path | sim code | what native `default` provider does | what we do |
|---|---|---|---|
| (a) target prefill | `TargetServer._calculate_batch_latency` (sim.py:1493) → `_estimate_batch_latency_from_provider` (1543) → `_make_phase_request("prefill")` (1568) | `prefill_latency_per_token × avg prompt length of the batch` (batch-size independent, no fixed cost) | `prefill_base_ms + prefill_ms_per_token × Σ prompt tokens in batch` |
| (b) target verify | same, `phase="decode"`; per-job fallback `_estimate_target_latency` (1519) | `decode_latency_per_token × max γ in batch` (batch-size independent, bonus position ignored) | `verify_base_ms + verify_ms_per_seq·B + verify_ms_per_token·Σ(γᵢ+1) + verify_ms_per_ctx_token·Σ contextᵢ` |
| (c) draft generation | `DraftServer._predict_generation_latency` (2285), called per round (3035 vanilla) or per beam (2536 eagle), with `target_id = draft id` | `decode_latency_per_token × γ` from the draft device's own YAML metadata (drafts are registered with the provider too, sim.py:5029) | `base_ms + ms_per_token × γ` from `calibration.json` → `draft.<vanilla\|specpp\|eagle>` |
| mixed prefill+verify batch | `_estimate_batch_latency_from_provider`: `max(prefill part, decode part)` | – | `target.mixed_batch_combine: max` (native) or `sum` |

The native provider cannot express a fixed per-step cost or any batch-size
dependence, so the shim installs its own provider
(`performance_model.type: dsdreal_calibrated`). The target's batch-latency
method is monkeypatched so that the combine rule and lockstep mode (below)
work. Each device's YAML entry carries a copy of its calibration block
(`dsdreal_calib`), so a generated `config.yaml` is self-describing.

Placeholder values (`calibration.json`): prefill 20 ms + 0.15 ms/token;
verify 25 ms + 0.10 ms per verified position (B=1, γ=4 → 25.5 ms;
B=64 → 57 ms); 1B draft 6 ms/token; EAGLE3 head 2.5 ms/token.

### Calibrating

Replace the placeholders with values measured on the A6000, then re-run
`run_sweep.py --force`:

* `target.prefill_*`: time vLLM prefill-only steps at a few prompt
  lengths and batch sizes, then fit `base + k·Σtokens`.
* `target.verify_*`: time target steps with speculation. With our
  proposer, one engine step = verify + `propose()` (network + draft). The
  round records carry `t_send_ms`, `t_recv_ms`, and `draft_ms`. Step
  compute ≈ (gap between consecutive `propose` calls) − (t_recv − t_send).
  Regress it on B and Σ(γ+1). Put any context-length slope in
  `verify_ms_per_ctx_token`.
* `draft.*.ms_per_token`: `SpeculateReply.draft_ms / γ` at batch 1.
  `ms_per_token_per_seq` is the growth with drafter batch size (lockstep
  mode only).
* Acceptance: build the profile with `scripts/build_acceptance_profile.py`
  (static γ=8 calibration run), then pass `--acceptance-profile
  runs/…/profile.json`. The shim answers every dsdSim acceptance lookup
  from it instead of the joblib model. Do this for EAGLE3 too, because the
  joblib model is the 1B drafter's.

## Approximations and mismatches (read before comparing numbers)

Items marked **[major]** can move the comparison by more than
calibration error.

1. **[major] Execution model: async (sim) vs. lockstep (ours).** In
   dsdSim every request runs its own draft → send → queue → verify →
   return loop, and the target keeps serving other requests while a
   request is drafting or on the wire. Our `RemoteDraftProposer.propose()`
   is called inside the vLLM engine step and blocks on
   `client.speculate()`, so each target step = verify(batch) + RTT + the
   slowest drafter's draft time, with all running requests in lockstep.
   `--mode lockstep` emulates this in the sim: link latency and draft
   latency are set to 0, and `rtt_ms + draft(γ_max, requests on the busiest
   drafter)` is added to every target batch. `results/` is the sim's own
   (async) model. `results/lockstep/` is the emulation. At placeholder
   calibration they differ by up to ~50 ms in TTFT.
2. **[major] Acceptance model.** `llama3_1b_vs_8b_500.joblib` gives
   first-position acceptance of only about 0.28–0.42, and its features are
   only (context_length, position). It was trained on logs where the
   drafter *sampled* (temperature/top-p) and the verifier took argmax
   (`acceptance/speculative_profiler.py`). The result is an accept ratio
   of about 0.19 and 0.73–0.77 accepted tokens per γ=4 round. Greedy
   1B→8B on GSM8K should be far higher. Positions ≥ 4 repeat position 4's
   probability (the classifier was trained with k=4). Use
   `--acceptance-profile` with our measured profile for a meaningful
   comparison.
3. **[major] SpecDec++ degenerates to γ=2.** Under the joblib model,
   cumulative rejection passes 0.7 at position 2 for every context, so
   `SpecPPGammaPolicy` (chosen once per conversation from the prompt
   length) always returns `min_gamma` = 2 (diagnostics: mean γ 2.00).
4. **[major] Conversation length semantics.** A dsdSim vanilla/SpecDec++
   conversation runs exactly `ceil(target_tokens/γ)` rounds, **counting
   drafted tokens, not accepted ones** (sim.py:2921, 3025). It then ends
   with only the accepted tokens as output (diagnostics "output/target
   tokens" ≈ 0.19). Our real runs generate `target_tokens` output tokens,
   which takes about `target_tokens / L̄` rounds (L̄ = accepted + bonus per
   round). Per-round metrics (TPOT, TTFT) are unaffected, but work per
   request, load, and saturation are. Use `--target-tokens-scale γ/L̄_real`
   to match round counts. The effect also depends on γ: SpecDec++ with
   larger γ gets fewer rounds regardless of acceptance.
5. **[major] Sim EAGLE model is unusable here.** `_run_eagle_conversation`
   ends the whole conversation at the first verify with 0 accepted tokens
   (sim.py:2669-2673, `conversation_completed = True`). With beam_width 1
   and ~0.3 first-token acceptance, requests stop after 1–2 rounds
   (output/target ≈ 0.01), so its TTFT/TPOT/throughput describe almost no
   decoding. It also uses the 1B-vs-8B acceptance model rather than an
   EAGLE3 one. It verifies beam branches sequentially (irrelevant at beam 1),
   and it enqueues decode jobs *before* the forward network delay
   (sim.py:2585-2607), so the draft→target hop overlaps queueing. We report
   it as required (`eagle3`) but recommend the `eagle3_chain` proxy (vanilla
   chain drafting, γ=4, EAGLE3-head draft latency) plus an EAGLE3
   acceptance profile.
6. **Arrival process.** dsdSim is closed-loop by design and refuses
   `think_time.enabled: false` (sim.py:4684). We use trace replay with
   rewritten `arrival_ms` (Poisson, same generator and seed as the real
   driver) and a constant 0 ms think time. That turns replay open-loop, but
   only if each drafter device holds one request. A `DraftServer` runs its
   conversations **one at a time**, and TTFT is measured from when it
   *starts* a conversation, not from the trace arrival (sim.py:2746-2753).
   A physical drafter with many requests would therefore queue them and
   hide that wait from TTFT. We give every request its own virtual draft
   device, so drafters never queue (and, in async mode, never batch or
   contend). The physical drafter index only labels requests (RR over N).
7. **Batching.** dsdSim has two batching stages: a per-phase
   scheduler, then the target window, each capped at 64 jobs. There is no
   admission limit on concurrent *requests*, as vLLM's max_num_seqs has,
   and no KV-cache model beyond a 1M-token admission counter. The target
   window is 0.01 ms rather than 0.
8. **No bonus token anywhere** (sim does not model it; accepted ≤ γ).
9. **Draft cost.** Async mode: per-request draft latency, no batching or
   drafter-GPU contention, and no draft prefill. Lockstep mode:
   `ms_per_token_per_seq` models contention on the busiest drafter.
10. **Network**: pure 5 ms delay each way per message. Payload bytes are
    ignored (bandwidth 0), so EAGLE3 hidden-state transfer (~24 KB/token) is
    free.

### Metric definitions (dsdSim `Metrics.summary`, sim.py:893, and `_collect_metrics_json`, sim.py:5259)

* **TTFT** = time from conversation start (= arrival here) to the end
  (response received) of the first verify round with ≥ 1 accepted token.
  It includes prefill and every zero-accept round before that.
* **TPOT (pooled)** = for every round with `a > 0` accepted, push `a`
  samples of `round_time/a`, then take the mean over all samples of all
  requests. This equals Σ(round time of rounds with a>0) / Σa. **Zero-accept
  rounds contribute no time.** (Vanilla/SpecDec++ samples are pushed twice,
  once from the conversation record and once from the job records. The
  mean is unchanged.)
* **Throughput (req/s)** = `conversation_throughput_rps` = completed
  conversations / (last target-job finish − first target-job creation). No
  burn-in. `throughput_jobs_s`, the sim's headline number, counts prefill +
  verify jobs and is **not** req/s.
* **accept ratio** = Σ accepted / Σ drafted (no bonus).

`dsdreal/metrics/collector.py`'s `sim` block uses the same rules. One
difference: our span starts at the first *arrival*, while dsdSim's starts
at the first prefill job reaching the target (arrival + 5 ms).

Each result JSON also contains `diagnostics`: mean γ, accepted/round,
output/target ratio, target utilization, jobs per batch, TTFT/TPOT
percentiles, dsdSim's ttft/decode breakdowns, and `requests`: one record
per request with the `validate.py::load_actual_run` field names, plus
`target_tokens` and `rounds`.

### Shim details (all in `dsdsim_shim.py`)

* Stub module for `performance.vidur_realtime`, so the sim imports without
  VIDUR.
* `src.sim` alias: `acceptance/regressor.py` does `from src import sim`.
  The alias stops it from importing the module a second time.
* Provider factory wrapper that adds `type: dsdreal_calibrated`.
* `TargetServer._estimate_batch_latency_from_provider` replaced: combine
  rule, plus the lockstep extra.
* `DraftServer._lookup_acceptance` wrapped (only active when
  `acceptance.dsdreal_profile` is set).
* The acceptance RandomForest's `n_jobs` is set from −1 to 1 after loading:
  identical predictions, ~6× faster. `pending_bucket`/`queue_bucket` are
  set to 1e9 because the classifier does not use those features and they
  only defeated the cache.

## Placeholder-calibrated results (async = dsdSim's own model; seed 0)

See `results/table3_sim.md` (and `results/lockstep/table3_sim.md`).

| Method | Metric | N=1 | N=2 | N=4 | N=7 |
|---|---|---|---|---|---|
| Vanilla SD | TTFT (ms) | 231.5 | 223.2 | 242.2 | 253.5 |
|  | TPOT (ms) | 28.7 | 31.5 | 33.6 | 35.3 |
|  | Thr. (req/s) | 0.86 | 1.85 | 3.71 | 6.35 |
|  | accept ratio | 0.185 | 0.190 | 0.195 | 0.195 |
| SpecDec++ | TTFT (ms) | 192.9 | 190.3 | 201.0 | 213.1 |
|  | TPOT (ms) | 32.5 | 33.9 | 35.7 | 38.4 |
|  | Thr. (req/s) | 0.86 | 1.83 | 3.66 | 6.19 |
|  | accept ratio | 0.270 | 0.272 | 0.273 | 0.273 |
| EAGLE3 (sim `eagle`, see mismatch 5) | TTFT (ms) | 82.6 | 84.8 | 89.9 | 95.6 |
|  | TPOT (ms) | 12.2 | 12.8 | 13.7 | 14.8 |
|  | Thr. (req/s) | 0.87 | 1.90 | 3.83 | 6.70 |
|  | accept ratio | 0.240 | 0.244 | 0.235 | 0.240 |
| EAGLE3 (chain proxy) | TTFT (ms) | 172.3 | 198.0 | 196.8 | 201.2 |
|  | TPOT (ms) | 22.6 | 23.4 | 24.6 | 26.6 |
|  | Thr. (req/s) | 0.87 | 1.87 | 3.74 | 6.44 |
|  | accept ratio | 0.189 | 0.194 | 0.198 | 0.194 |

The offered load is `per_draft_rps × N` = 1, 2, 4, 7 req/s, and nothing
saturates at placeholder latencies. Throughput tracks the arrival rate
(slightly below it, because the span includes the drain tail), and every
request completes. This does **not** show the "rise then saturate" shape;
that depends on the real verify/draft costs and on mismatch 4.

## Calibrating from a real run (automated)

`sim/calibrate_from_run.py` fits the target-verify and draft constants from a
real run's `steps.jsonl` (rows of one `propose()` call share `t_send_ms`):

- verify step = next call's `t_send` − this call's `t_recv`, regressed on
  batch size B and Σ(γᵢ+1); steps that also prefilled a new sequence are
  dropped (prefill terms are separate);
- draft round = the drafter's own `draft_ms` for the call's batch, regressed
  on γ and B (if γ never varied, base is fixed at 0 — intercept and slope are
  the same column then).

```bash
python sim/calibrate_from_run.py runs/table3/calib_g8 --method vanilla -o sim/calibration.measured.json
python sim/calibrate_from_run.py runs/table3/specpp_N1 --method specpp \
    --base sim/calibration.measured.json -o sim/calibration.measured.json
python sim/calibrate_from_run.py runs/table3/eagle3_N1 --method eagle \
    --base sim/calibration.measured.json -o sim/calibration.measured.json
$SIMPY sim/run_sweep.py --calibration sim/calibration.measured.json \
    --acceptance-profile runs/table3/calib_g8/profile.json --mode lockstep --out sim/results/measured
```

It prints R² and the 90th-percentile absolute residual for each fit; check
them before trusting the simulator's numbers. Prefill terms are not fitted
yet (keep the placeholders or measure them from a no-draft baseline run).
