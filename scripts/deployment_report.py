#!/usr/bin/env python
"""Per-deployment breakdown of real runs: where each component ran and where
the time of a step goes.

    python scripts/deployment_report.py runs/table3 > runs/table3/deployment.md

For every <method>_N<n> run: GPU placement (placement.json from
scripts/run_nto1.sh), and per engine step (one propose() call; rows of a call
share t_send_ms):
  - propose wall  = drafts back - sent (network + the slowest drafter),
  - draft (max)   = the slowest drafter's own compute that step,
  - network+queue = propose wall - draft (max) (injected RTT + transport),
  - verify step   = next propose - this call's drafts back (target forward),
  - batch B       = sequences in the step,
and per drafter: sequences served, rounds, mean draft ms, acceptance ratio.
"""
from __future__ import annotations

import json
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path


def p50(xs):
    return st.median(xs) if xs else float("nan")


def p90(xs):
    xs = sorted(xs)
    return xs[int(0.9 * (len(xs) - 1))] if xs else float("nan")


def run_report(run: Path) -> list[str]:
    rows = [json.loads(line) for line in (run / "steps.jsonl").read_text().split("\n") if line.strip()]
    calls = defaultdict(list)
    for r in rows:
        calls[round(r["t_send_ms"], 3)].append(r)
    ks = sorted(calls)
    wall = [calls[k][0]["t_recv_ms"] - k for k in ks]
    dmax = [max(r["draft_ms"] for r in calls[k]) for k in ks]
    net = [w - d for w, d in zip(wall, dmax)]
    ver = [calls[ks[i + 1]][0]["t_send_ms"] - calls[ks[i]][0]["t_recv_ms"] for i in range(len(ks) - 1)]
    B = [len(calls[k]) for k in ks]

    place = {}
    if (run / "placement.json").exists():
        place = json.loads((run / "placement.json").read_text())
    out = [f"### {run.name}", ""]
    if place:
        by_gpu = defaultdict(list)
        for d, info in place["drafters"].items():
            by_gpu[info["gpu"]].append(d)
        gpus = ", ".join(f"GPU{g}: {' '.join(ds)}" for g, ds in sorted(by_gpu.items()))
        out.append(f"placement ({place.get('host', '')}): target GPU{place['target']['gpu']}; "
                   f"drafters {gpus} -> tier {place['tier']}")
        out.append("")
    out += ["| per step | p50 | p90 |", "|---|---|---|",
            f"| batch B | {p50(B):.0f} | {p90(B):.0f} |",
            f"| propose wall (ms) | {p50(wall):.1f} | {p90(wall):.1f} |",
            f"| draft, slowest drafter (ms) | {p50(dmax):.1f} | {p90(dmax):.1f} |",
            f"| network + transport (ms) | {p50(net):.1f} | {p90(net):.1f} |",
            f"| verify step (ms) | {p50(ver):.1f} | {p90(ver):.1f} |", ""]

    per = defaultdict(lambda: {"seqs": set(), "rounds": 0, "draft": [], "acc": 0, "drafted": 0})
    for r in rows:
        d = per[r["drafter_id"]]
        d["seqs"].add(r["request_id"])
        d["rounds"] += 1
        d["draft"].append(r["draft_ms"])
        d["acc"] += r["accepted"]
        d["drafted"] += r["n_draft"]
    gpu_of = {d: info["gpu"] for d, info in place.get("drafters", {}).items()}
    out += ["| drafter | GPU | requests | rounds | draft ms p50 | accept ratio |", "|---|---|---|---|---|---|"]
    for d in sorted(per, key=lambda x: int(re.sub(r"\D", "", x) or 0)):
        v = per[d]
        out.append(f"| {d} | {gpu_of.get(d, '?')} | {len(v['seqs'])} | {v['rounds']} | "
                   f"{p50(v['draft']):.1f} | {v['acc'] / max(v['drafted'], 1):.3f} |")
    return out + [""]


def main(root: str) -> int:
    runs = sorted((p.parent for p in Path(root).glob("*_N*/steps.jsonl")),
                  key=lambda p: (p.name.split("_N")[0], int(p.name.split("_N")[1])))
    for run in runs:
        print("\n".join(run_report(run)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "runs/table3"))
