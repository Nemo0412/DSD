#!/usr/bin/env python
"""Generate a dsdSim YAML config (+ retimed trace) mirroring one dsd-real run.

    python sim/make_config.py --method vanilla --n-drafters 4 \
        --calibration sim/calibration.json --out-dir sim/configs/vanilla_N4

Writes <out-dir>/config.yaml and <out-dir>/trace.jsonl. Pure stdlib + PyYAML
(no dsdSim import needed), so it runs in either venv.

Topology written:
  * 1 target `t0` (Llama-3.1-8B, batch cap = max_num_seqs).
  * One *virtual* drafter device per request, id `d{p}_r{i}`, where p = i % N
    is the physical drafter the request is round-robined to (our RR routing).
    dsdSim's DraftServer runs one conversation at a time, so giving every
    request its own device is the only way to let a drafter serve many
    concurrent requests without hiding queueing time outside TTFT (see
    README, "Arrival process").
  * Each virtual drafter has a draft->target link of rtt/2 and target->draft
    of rtt/2 (pure delay element: no bandwidth term, no jitter, unbounded
    link concurrency).
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import yaml

METHODS = ("vanilla", "specpp", "eagle")
ACCEPTANCE_JOBLIB = "src/acceptance/llama3_1b_vs_8b_500.joblib"


def poisson_arrivals(n: int, rate_rps: float, seed: int = 0) -> list[float]:
    """Byte-for-byte the same generator as dsdreal/workload/trace.py
    (random.Random(seed), exponential gaps, first arrival at 0), so the sim
    replays exactly the arrival times the real driver uses."""
    if rate_rps <= 0:
        raise ValueError("rate_rps must be positive")
    rng = random.Random(seed)
    t, out = 0.0, []
    for _ in range(n):
        out.append(t)
        t += rng.expovariate(rate_rps) * 1000.0
    return out


def load_trace(path: Path) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    rows.sort(key=lambda d: float(d["arrival_ms"]))  # as dsdreal load_sim_trace
    return rows


def default_n_requests(n: int) -> int:
    return min(400, 120 * n)


def load_replay(run_dir: str | Path) -> dict[str, dict]:
    """request_id -> {"gamma": int, "accepted": [per verified round]} from a
    real dsd-real run (steps.jsonl: one row per verified round, `i` = round
    index, accepted excludes the bonus token -- dsdSim's own convention)."""
    per: dict[str, list[tuple[int, int, int]]] = {}
    for line in (Path(run_dir) / "steps.jsonl").read_text().split("\n"):
        if line.strip():
            d = json.loads(line)
            per.setdefault(d["request_id"], []).append((int(d["i"]), int(d["gamma"]), int(d["accepted"])))
    out = {}
    for rid, rounds in per.items():
        rounds.sort()
        out[rid] = {"gamma": rounds[0][1], "accepted": [a for _, _, a in rounds]}
    return out


def build(args) -> tuple[dict, list[dict]]:
    root = Path(args.dsdsim_root or os.environ.get("DSDSIM_ROOT", "")).resolve()
    calib_all = json.loads(Path(args.calibration).read_text())
    n = args.n_drafters
    n_req = args.n_requests or default_n_requests(n)
    trace_src = Path(args.trace) if args.trace else root / "traces" / "gsm8k_trace_master.jsonl"
    rows = load_trace(trace_src)[:n_req]
    if len(rows) < n_req:
        raise SystemExit(f"trace {trace_src} has only {len(rows)} rows (< {n_req})")
    rate = args.per_draft_rps * n
    arrivals = poisson_arrivals(len(rows), rate, args.seed)

    replay = load_replay(args.replay_from) if args.replay_from else {}
    replay_by_draft: dict[str, dict] = {}
    trace = []
    for i, (r, t) in enumerate(zip(rows, arrivals)):
        rid, did = str(r["request_id"]), f"d{i % n}_r{i:04d}"
        target = max(1, round(int(r["target_tokens"]) * args.target_tokens_scale))
        if rid in replay:
            # dsdSim runs ceil(answer / gamma) rounds per conversation: make
            # that exactly the real run's number of verified rounds.
            rp = replay[rid]
            target = len(rp["accepted"]) * rp["gamma"]
            replay_by_draft[did] = rp
        trace.append({
            "request_id": rid,
            "arrival_ms": round(t, 6),
            "prompt_tokens": int(r["prompt_tokens"]),
            "target_tokens": target,
            "draft_id": did,
            "device_tier": "default",
        })

    draft_kind = args.draft_profile or {"vanilla": "vanilla", "specpp": "specpp", "eagle": "eagle"}[args.method]
    calib = {
        "mode": args.mode,
        "rtt_ms": args.rtt_ms,
        "n_drafters": n,
        "target": {k: v for k, v in calib_all["target"].items() if not k.startswith("_")},
        "draft": {k: v for k, v in calib_all["draft"][draft_kind].items() if not k.startswith("_")},
        "source": str(Path(args.calibration).resolve()),
        "status": calib_all.get("_status", ""),
    }
    sched = calib_all.get("scheduling", {})
    one_way = 0.0 if args.mode == "lockstep" else args.rtt_ms / 2.0

    devices = [{
        "id": "t0",
        "role": "target",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "gpu": "RTX_A6000",
        "batch_size": args.max_num_seqs,
        "batch_window_ms": float(sched.get("target_batch_window_ms", 0.01)),
        "cluster": "default",
        "dsdreal_calib": calib,
    }]
    connections = []
    draft_model = ("yuhuili/EAGLE3-LLaMA3.1-Instruct-8B" if args.method == "eagle"
                   else "meta-llama/Llama-3.2-1B-Instruct")
    for rec in trace:
        did = rec["draft_id"]
        devices.append({
            "id": did,
            "role": "draft",
            "model": draft_model,
            "gpu": f"drafter_{did.split('_r')[0]}",
            "capability": 1.0,
            "cluster": "default",
            "dsdreal_calib": calib,
        })
        connections.append({
            "draft": did,
            "target": "t0",
            "forward_latency_ms": one_way,
            "response_latency_ms": one_way,
            "jitter_pct": 0.0,
        })

    last_arrival = trace[-1]["arrival_ms"]
    spec: dict = {
        "framework": "eagle" if args.method == "eagle" else "vanilla",
        "execution_mode": "distributed",
        "acceptance": {
            "model": str(root / ACCEPTANCE_JOBLIB),
            "use_classifier": True,
            "default_rate": 0.7,
            "context_bucket": 1,
            "depth_bucket": 1,
            # The classifier's features are (context_length, position) only;
            # pending/queue are just cache-key noise. Bucketing them to 0
            # makes the caches hit without changing any probability.
            "pending_bucket": 1_000_000_000,
            "queue_bucket": 1_000_000_000,
        },
    }
    if args.acceptance_profile:
        # joblib stays loaded (DraftServer requires a model object); the shim
        # answers every lookup from the measured profile instead.
        spec["acceptance"]["dsdreal_profile"] = str(Path(args.acceptance_profile).resolve())
    if replay_by_draft:
        # every conversation replays its real gamma and per-round accepted
        # counts (the shim; one virtual drafter per request = one sequence)
        rp_path = Path(args.out_dir).resolve() / "replay.json"
        rp_path.parent.mkdir(parents=True, exist_ok=True)
        rp_path.write_text(json.dumps(replay_by_draft))
        spec["acceptance"]["dsdreal_replay"] = str(rp_path)
    if args.acceptance_fixed_rate is not None:
        spec["acceptance"] = {"disable_model": True, "default_rate": args.acceptance_fixed_rate}
    if args.method == "specpp":
        spec["gamma_policy"] = {"type": "specpp", "min_gamma": 2, "max_gamma": 8, "stop_threshold": 0.7}
    else:
        spec["gamma_policy"] = {"type": "constant"}
    if args.method == "eagle":
        spec["eagle"] = {
            "beam_width": args.eagle_beam_width,
            "max_depth": args.eagle_max_depth,
            "min_depth": 1,
            "prune_score_threshold": 0.0,
            "load_control": {"enabled": False},
        }

    wait = float(sched.get("scheduler_max_wait_ms", 0.0))
    lane = {"pool": "default", "queue_policy": "fifo", "max_batch_requests": args.max_num_seqs,
            "max_wait_ms": wait, "max_queue_depth": 0}
    cfg = {
        "_dsdreal": {
            "method": args.method, "n_drafters": n, "n_requests": n_req, "per_draft_rps": args.per_draft_rps,
            "total_rate_rps": rate, "rtt_ms": args.rtt_ms, "seed": args.seed, "mode": args.mode,
            "client_side": args.client_side,
            "trace_source": str(trace_src), "calibration": str(Path(args.calibration).resolve()),
            "draft_profile": draft_kind, "target_tokens_scale": args.target_tokens_scale,
            "replay_from": str(Path(args.replay_from).resolve()) if args.replay_from else None,
            "acceptance": ("fixed:%s" % args.acceptance_fixed_rate if args.acceptance_fixed_rate is not None
                           else "profile:%s" % args.acceptance_profile if args.acceptance_profile
                           else ACCEPTANCE_JOBLIB),
        },
        "sim_time_ms": float(last_arrival + args.tail_ms),
        "seed": args.seed,
        "verbose": False,
        "debug": False,
        "execution_mode": "blocking",
        "gamma": args.gamma if args.method != "eagle" else args.eagle_max_depth,
        "speculation": spec,
        "trace_path": str(Path(args.out_dir).resolve() / "trace.jsonl"),
        "trace_defaults": {},
        "trace_replay": {},
        "router": "round_robin",
        "router_params": {},
        "global_router": "disabled",
        "workload": {"arrival": "poisson", "rate_rps": rate},
        # dsdSim refuses to run with think time disabled; a constant 0 ms think
        # time turns trace replay into open-loop arrivals.
        "think_time": {"enabled": True, "distribution": "constant", "mean_ms": 0.0, "min_ms": 0.0},
        "network": {"enabled": True, "bandwidth_mbps": 0.0, "link_capacity": 1_000_000, "jitter_pct": 0.0},
        "performance_model": {"type": "dsdreal_calibrated"},
        "mixed_batching": True,
        "burn_in_ms": 0.0,
        "scheduler": {"type": "baseline", "prefill": dict(lane), "decode": dict(lane),
                      "kv": {"default_capacity_tokens": 1_000_000}},
        "devices": devices,
        "connections": connections,
    }
    return cfg, trace


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=METHODS, required=True)
    ap.add_argument("--n-drafters", type=int, required=True)
    ap.add_argument("--calibration", default=str(Path(__file__).with_name("calibration.json")))
    ap.add_argument("--per-draft-rps", type=float, default=1.0)
    ap.add_argument("--rtt-ms", type=float, default=10.0)
    ap.add_argument("--n-requests", type=int, default=None, help="default min(400, 120*N)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=("async", "lockstep"), default="async",
                    help="async = dsdSim's native per-request pipeline; lockstep = RTT+drafting "
                         "charged inside every target step (emulates our blocking proposer)")
    ap.add_argument("--gamma", type=int, default=4, help="static gamma (vanilla); SpecDec++ fallback")
    ap.add_argument("--eagle-beam-width", type=int, default=1)
    ap.add_argument("--eagle-max-depth", type=int, default=4)
    ap.add_argument("--acceptance-fixed-rate", type=float, default=None,
                    help="bypass the joblib regressor: constant per-position acceptance")
    ap.add_argument("--acceptance-profile", default=None,
                    help="dsd-real AcceptanceProfile JSON (scripts/build_acceptance_profile.py) "
                         "to use instead of the joblib classifier (needs the shim)")
    ap.add_argument("--draft-profile", default=None,
                    help="key of calibration.json 'draft' to use (default: the method's own); "
                         "e.g. --method vanilla --draft-profile eagle = chain drafting at EAGLE3-head cost")
    ap.add_argument("--target-tokens-scale", type=float, default=1.0,
                    help="multiply every trace target_tokens (dsdSim runs ceil(target/gamma) rounds; "
                         "gamma/L_real makes its round count match a real run with mean acceptance length L_real)")
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--replay-from", default=None, metavar="RUN_DIR",
                    help="replay a real run's per-request gamma and per-round accepted counts "
                         "(its steps.jsonl) instead of sampling acceptance")
    ap.add_argument("--client-side", action="store_true",
                    help="requests born at the drafters' clients (dsdSim's own scenario); see run_sim.py")
    ap.add_argument("--tail-ms", type=float, default=600_000.0,
                    help="sim horizon after the last arrival (must cover the slowest request)")
    ap.add_argument("--trace", default=None, help="default $DSDSIM_ROOT/traces/gsm8k_trace_master.jsonl")
    ap.add_argument("--dsdsim-root", default=None, help="default $DSDSIM_ROOT")
    ap.add_argument("--out-dir", required=True)
    return ap.parse_args(argv)


def write(args) -> Path:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg, trace = build(args)
    (out / "trace.jsonl").write_text("".join(json.dumps(r) + "\n" for r in trace))
    with (out / "config.yaml").open("w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False, width=120)
    return out / "config.yaml"


if __name__ == "__main__":
    print(write(parse()))
