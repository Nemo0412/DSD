#!/usr/bin/env python
"""Run ONE dsdSim config through the shim and write a result JSON.

    DSDSIM_ROOT=/path/to/dsdSim python sim/run_sim.py CONFIG.yaml -o result.json [--log sim.log]

Must run in the sim venv (simpy, numpy, scikit-learn==1.5.2, joblib, pyyaml).
The simulator is chatty on stdout; it is redirected to --log.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dsdsim_shim  # noqa: E402


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def run(config: Path, log_path: Path | None, dsdsim_root: str | None) -> dict:
    import yaml

    raw = yaml.safe_load(config.read_text())
    meta = dict(raw.get("_dsdreal", {}))
    t0 = time.perf_counter()
    log_cm = open(log_path, "w") if log_path else open("/dev/null", "w")
    with log_cm as log, contextlib.redirect_stdout(log):
        sim = dsdsim_shim.load_sim(dsdsim_root)
        cfg = sim.load_config_from_mapping(raw)
        metrics, targets, drafts = sim.run(cfg)
        summary = metrics.summary()
        mj = sim._collect_metrics_json(cfg, metrics, summary, targets)
    wall_s = time.perf_counter() - t0

    # ---- diagnostics from the raw job / conversation records -------------
    trace = [json.loads(l) for l in Path(cfg.trace_path).read_text().splitlines() if l.strip()]
    by_id = {r["request_id"]: r for r in trace}
    convs = [c for c in metrics.conversations if c.get("completed")]
    decode_jobs = defaultdict(list)
    for j in metrics.completed:
        if j.job_type == "decode" and j.request_id:
            decode_jobs[j.request_id].append(j)
    rounds = sum(len(v) for v in decode_jobs.values())
    drafted = sum(j.token_count for v in decode_jobs.values() for j in v)
    accepted = sum(int(c.get("tokens_accepted", 0)) for c in convs)
    gammas = [j.gamma_override if j.gamma_override > 0 else j.token_count
              for v in decode_jobs.values() for j in v]
    first_gamma = [v[0].gamma_override or v[0].token_count for v in decode_jobs.values() if v]
    out_ratio = [c["tokens_accepted"] / c["answer_tokens"] for c in convs if c.get("answer_tokens")]
    t = targets[0] if targets else None

    requests = []
    for c in convs:
        r = by_id.get(c["id"], {})
        requests.append({  # dsdSim verification/validate.py::load_actual_run field names
            "request_id": c["id"],
            "arrival_ms": r.get("arrival_ms", c["start_ms"]),
            "prompt_tokens": r.get("prompt_tokens"),
            "completion_tokens": c.get("tokens_accepted"),
            "total_latency_ms": c["end_ms"] - c["start_ms"],
            "ttft_ms": c.get("ttft_ms"),
            "draft_id": c.get("draft_id", "").split("_r")[0],
            "target_tokens": r.get("target_tokens"),
            "rounds": len(decode_jobs.get(c["id"], [])),
        })

    # Client-side scenario in lockstep mode: lockstep zeroes the links and
    # charges RTT inside target steps, which drops the two hops a client-born
    # request pays outside any step -- prompt upload and first-token return.
    # Links here are pure delays (no bandwidth term, no link queueing) and
    # arrivals are open-loop, so those hops shift every request by exactly
    # rtt/2 each way: TTFT and end-to-end latency grow by rtt_ms, TPOT and
    # throughput are unchanged. (Async mode simulates the hops itself.)
    offset = (float(meta.get("rtt_ms", 0.0))
              if meta.get("client_side") and meta.get("mode") == "lockstep" else 0.0)
    if offset:
        for r in requests:
            r["ttft_ms"] = None if r["ttft_ms"] is None else r["ttft_ms"] + offset
            r["total_latency_ms"] += offset
    table = {
        "ttft_mean_ms": (mj.get("ttft_avg_ms") or 0.0) + offset if mj.get("ttft_avg_ms") is not None else None,
        "tpot_pooled_ms": mj.get("tpot_avg_ms"),
        "throughput_rps": mj.get("conversation_throughput_rps"),
        "acceptance_ratio": mj.get("acceptance_rate"),
    }
    diag = {
        "n_requests_in_trace": len(trace),
        "completed_conversations": len(convs),
        "all_completed": len(convs) == len(trace),
        "decode_rounds": rounds,
        "mean_gamma_per_round": _mean(gammas),
        "mean_gamma_first_round": _mean(first_gamma),
        "mean_accepted_per_round_no_bonus": accepted / rounds if rounds else None,
        "drafted_tokens": drafted,
        "accepted_tokens": accepted,
        "mean_output_over_target_tokens": _mean(out_ratio),
        "target_total_batches": t.total_batches if t else None,
        "target_avg_batch_items": (t.total_batch_items / t.total_batches) if t and t.total_batches else None,
        "target_busy_ms": t.busy_ms if t else None,
        "observed_span_ms": mj.get("observed_span_ms"),
        "target_utilization": (t.busy_ms / mj["observed_span_ms"]) if t and mj.get("observed_span_ms") else None,
        "ttft_p50_ms": mj.get("ttft_p50_ms"), "ttft_p95_ms": mj.get("ttft_p95_ms"),
        "tpot_p50_ms": mj.get("tpot_p50_ms"), "tpot_p95_ms": mj.get("tpot_p95_ms"),
        "sim_wall_s": wall_s,
        "client_side_ttft_offset_ms": offset,
    }
    for k, v in mj.items():
        if k.startswith("ttft_breakdown_") or k.startswith("decode_breakdown_"):
            diag[k] = v
    return {"config": meta, "config_path": str(config.resolve()), "table": table,
            "diagnostics": diag, "sim_metrics_json": {k: v for k, v in mj.items() if k != "profiler"},
            "requests": requests}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--log", type=Path, default=None)
    ap.add_argument("--dsdsim-root", default=None)
    a = ap.parse_args(argv)
    res = run(a.config, a.log, a.dsdsim_root)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2, default=float))
    tb, dg = res["table"], res["diagnostics"]
    print(f"{a.config.parent.name}: TTFT {tb['ttft_mean_ms']:.1f} ms  TPOT {tb['tpot_pooled_ms']:.1f} ms  "
          f"thr {tb['throughput_rps']:.3f} req/s  acc {tb['acceptance_ratio']:.3f}  "
          f"completed {dg['completed_conversations']}/{dg['n_requests_in_trace']}  ({dg['sim_wall_s']:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
