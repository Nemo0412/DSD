#!/usr/bin/env python3
"""Run full DSD-Sim WAN RTT sensitivity experiments."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import statistics
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sim import load_config_from_mapping, simulate_config_obj  # noqa: E402


METRICS = {
    "ttft_avg_ms": ("ttft_avg_ms", "avg_ttft_ms"),
    "tpot_avg_ms": ("tpot_avg_ms", "avg_tpot_ms"),
    "throughput_rps": (
        "conversation_throughput_rps",
        "goodput_rps",
        "throughput_jobs_s",
    ),
}


def _metric(metrics: dict[str, Any], aliases: tuple[str, ...]) -> float:
    for key in aliases:
        value = metrics.get(key)
        if value is not None:
            return float(value)
    return 0.0


def _prepare_config(
    base: dict[str, Any],
    *,
    trace_path: Path,
    seed: int,
    rtt_ms: float,
    total_nodes: int,
) -> dict[str, Any]:
    cfg = deepcopy(base)
    cfg["trace_path"] = str(trace_path)
    cfg["seed"] = seed
    cfg["verbose"] = False
    cfg["debug"] = False

    clusters = cfg.get("auto_topology", {}).get("clusters", [])
    if len(clusters) != 1:
        raise ValueError("WAN sweep expects exactly one auto_topology cluster")
    cluster = clusters[0]

    target_count = max(1, round(total_nodes / 5))
    draft_count = total_nodes - target_count
    targets = cluster["targets"]
    drafts = cluster["drafts"]
    targets["count"] = target_count
    tiers = targets.get("tiers", [])
    if len(tiers) != 1 or not tiers[0].get("name"):
        raise ValueError("WAN sweep expects exactly one named target tier")
    tiers[0]["count"] = target_count
    tier_name = str(tiers[0]["name"])

    drafts["count"] = draft_count
    counts = drafts.get("count_by_label")
    if counts:
        if len(counts) != 1:
            raise ValueError("WAN sweep expects one draft label")
        counts[next(iter(counts))] = draft_count

    # DSD-Sim stores forward and response delays separately. Configure each
    # direction to half of the requested end-to-end round-trip time.
    connectivity = cluster.setdefault("connectivity", {})
    connectivity.pop("network_model", None)
    one_way_ms = rtt_ms / 2.0
    connectivity["net_ms_ranges"] = {tier_name: [one_way_ms, one_way_ms]}
    connectivity["link_jitter_pct"] = 0.0
    return cfg


def _run(cfg: dict[str, Any]) -> dict[str, float]:
    config = load_config_from_mapping(cfg)
    config.verbose = False
    config.debug = False
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
        metrics = simulate_config_obj(config, emit_output=False)
    result = {name: _metric(metrics, aliases) for name, aliases in METRICS.items()}
    result["completed"] = float(metrics.get("completed_conversation_count") or 0.0)
    return result


def _mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "experiments/configs/acceptance_sensitivity_64.yaml",
    )
    parser.add_argument(
        "--trace",
        type=Path,
        default=REPO_ROOT / "traces/gsm8k_trace_10s_with_acceptance.jsonl",
    )
    parser.add_argument("--rtts", type=float, nargs="+", default=[5, 20, 50, 100, 150])
    parser.add_argument("--seeds", type=int, nargs="+", default=[121, 122, 123])
    parser.add_argument("--total-nodes", type=int, default=4096)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "experiments/results/wan_rtt_sensitivity.json",
    )
    args = parser.parse_args()

    base = yaml.safe_load(args.config.read_text()) or {}
    trace_path = args.trace.resolve()
    all_runs: list[dict[str, float]] = []
    started = time.time()

    for rtt_ms in args.rtts:
        for seed in args.seeds:
            cfg = _prepare_config(
                base,
                trace_path=trace_path,
                seed=seed,
                rtt_ms=rtt_ms,
                total_nodes=args.total_nodes,
            )
            run_started = time.time()
            metrics = _run(cfg)
            row = {"rtt_ms": rtt_ms, "seed": seed, **metrics}
            all_runs.append(row)
            print(
                f"RTT={rtt_ms:g}ms seed={seed}: "
                f"TTFT={metrics['ttft_avg_ms']:.2f}ms "
                f"TPOT={metrics['tpot_avg_ms']:.2f}ms "
                f"throughput={metrics['throughput_rps']:.2f}req/s "
                f"({time.time() - run_started:.1f}s)",
                flush=True,
            )

    summary = []
    for rtt_ms in args.rtts:
        runs = [row for row in all_runs if row["rtt_ms"] == rtt_ms]
        summary.append(
            {
                "rtt_ms": rtt_ms,
                **{
                    name: _mean_std([row[name] for row in runs])
                    for name in ("ttft_avg_ms", "tpot_avg_ms", "throughput_rps")
                },
            }
        )

    payload = {
        "method": "full discrete-event re-simulation",
        "config": str(args.config),
        "trace": str(trace_path),
        "total_nodes": args.total_nodes,
        "draft_nodes": args.total_nodes - round(args.total_nodes / 5),
        "target_nodes": round(args.total_nodes / 5),
        "rtt_semantics": "end-to-end RTT; forward and response delays are RTT/2 each",
        "seeds": args.seeds,
        "elapsed_s": time.time() - started,
        "runs": all_runs,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
