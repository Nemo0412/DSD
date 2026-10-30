#!/usr/bin/env python
"""Print draft, verify, and end-to-end times from one dsd-real run."""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def median(xs):
    return statistics.median(xs) if xs else float("nan")


def mean(xs):
    return statistics.fmean(xs) if xs else float("nan")


def main() -> int:
    run = Path(sys.argv[1])
    rows = [json.loads(line) for line in (run / "steps.jsonl").read_text().splitlines() if line.strip()]
    calls = defaultdict(list)
    for row in rows:
        calls[round(row["t_send_ms"], 3)].append(row)
    keys = sorted(calls)
    draft = [max(r["draft_ms"] for r in calls[k]) for k in keys]
    verify = [
        calls[keys[i + 1]][0]["t_send_ms"] - calls[keys[i]][0]["t_recv_ms"]
        for i in range(len(keys) - 1)
    ]
    net = [calls[k][0]["t_recv_ms"] - k - max(r["draft_ms"] for r in calls[k]) for k in keys]
    metrics = json.loads((run / "dsd_metrics.json").read_text())
    e2e = metrics["system_metrics"]["standard"]["e2e_ms"]
    ttft = metrics["system_metrics"]["standard"]["ttft_ms"]
    tpot = metrics["system_metrics"]["standard"]["tpot_ms"]
    print(f"run {run}")
    print(f"steps {len(keys)}  requests {metrics['system_metrics']['completed']}")
    print(f"draft_ms    mean={mean(draft):.2f}  median={median(draft):.2f}  n={len(draft)}")
    print(f"verify_ms   mean={mean(verify):.2f}  median={median(verify):.2f}  n={len(verify)}")
    print(f"network_ms  mean={mean(net):.2f}  median={median(net):.2f}  n={len(net)}")
    print(f"e2e_ms      mean={e2e['mean']:.1f}  p50={e2e['p50']:.1f}")
    print(f"ttft_ms     mean={ttft['mean']:.1f}  p50={ttft['p50']:.1f}")
    print(f"tpot_ms     mean={tpot['mean']:.1f}  p50={tpot['p50']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
