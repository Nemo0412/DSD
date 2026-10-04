"""Write a run's results to disk.

- `dsd_metrics.json`: config, env, both metric conventions, per-request rows.
- `steps.jsonl`: one line per verified round -- the raw material.
- `actual_run.jsonl`: dsdSim's real-run format
  (src/verification/validate.py::load_actual_run), so the simulator's own
  `compare_runs` reads it directly. `ttft_ms` there is the sim convention;
  everything else rides in `metadata`, which that loader keeps verbatim.

All times are written relative to the run's t0 (ms).
"""
from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import time
from pathlib import Path

from dsdreal.metrics.collector import RequestMetrics, summarize

SCHEMA_VERSION = 1


def _clean(x):
    """JSON has no NaN; write null instead of a value readers choke on."""
    if isinstance(x, float) and math.isnan(x):
        return None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    return x


def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parent, timeout=5).stdout.strip() or None
    except Exception:
        return None


def request_row(m: RequestMetrics, t0_ms: float) -> dict:
    return {
        "request_id": m.request_id,
        "drafter_id": m.drafter_id,
        "arrival_ms": m.arrival_ms - t0_ms,
        "finish_ms": m.finish_ms - t0_ms,
        "prompt_len": m.prompt_len,
        "output_len": m.output_len,
        "ttft_ms": m.ttft_ms,
        "tpot_ms": m.tpot_ms,
        "e2e_ms": m.e2e_ms,
        "ttft_sim_ms": m.ttft_sim_ms,
        "tpot_sim_ms": m.tpot_sim_ms,
        "num_rounds": len(m.rounds),
        "num_accepted": m.num_accepted,
        "num_drafted": m.num_drafted,
        "acceptance_ratio": m.acceptance_ratio,
        "mean_acceptance_length": m.mean_acceptance_length,
        "accounting_ok": m.accounting_ok,
        "notes": m.notes,
    }


def actual_run_row(m: RequestMetrics, t0_ms: float) -> dict:
    return {
        "request_id": m.request_id,
        "arrival_ms": m.arrival_ms - t0_ms,
        "prompt_tokens": m.prompt_len,
        "completion_tokens": m.output_len,
        "total_latency_ms": m.e2e_ms,
        "ttft_ms": m.ttft_sim_ms,
        "draft_id": m.drafter_id,
        "metadata": {
            "ttft_ms_standard": m.ttft_ms,
            "tpot_ms_standard": m.tpot_ms,
            "tpot_ms_sim": m.tpot_sim_ms,
            "acceptance_ratio": m.acceptance_ratio,
            "num_rounds": len(m.rounds),
        },
    }


def step_rows(m: RequestMetrics, t0_ms: float):
    for i, r in enumerate(m.rounds):
        yield {
            "request_id": m.request_id, "i": i, "drafter_id": r.rec.drafter_id,
            "gamma": r.rec.gamma, "n_draft": r.rec.n_draft, "accepted": r.accepted,
            "delta": r.delta, "num_confirmed": r.rec.num_confirmed,
            "t_send_ms": r.rec.t_send_ms - t0_ms, "t_recv_ms": r.rec.t_recv_ms - t0_ms,
            "t_end_ms": r.t_end_ms - t0_ms, "draft_ms": r.rec.draft_ms,
            "injected_ms": r.rec.injected_ms,
            "net_excess_ms": (r.rec.t_recv_ms - r.rec.t_send_ms) - r.rec.injected_ms - r.rec.draft_ms,
        }


def write_run(out_dir: str | os.PathLike, per_request: list[RequestMetrics], t0_ms: float,
              config: dict, trim_frac: float = 0.1, extra: dict[str, dict] | None = None) -> dict:
    """`extra`: request_id -> fields merged into that request's row (e.g. the
    router's decision: target_id, queue_depth_at_decision)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    system = summarize(per_request, trim_frac=trim_frac)
    doc = {
        "schema_version": SCHEMA_VERSION,
        "run_id": out.name,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": config,
        "env": {"git": _git_sha(), "host": platform.node(), "python": platform.python_version()},
        "system_metrics": system,
        "per_request": [{**request_row(m, t0_ms), **(extra or {}).get(m.request_id, {})}
                        for m in per_request],
    }
    (out / "dsd_metrics.json").write_text(json.dumps(_clean(doc), indent=2))
    with open(out / "steps.jsonl", "w") as f:
        for m in per_request:
            for row in step_rows(m, t0_ms):
                f.write(json.dumps(_clean(row)) + "\n")
    with open(out / "actual_run.jsonl", "w") as f:
        for m in per_request:
            if not math.isnan(m.e2e_ms):
                f.write(json.dumps(_clean(actual_run_row(m, t0_ms))) + "\n")
    return system
