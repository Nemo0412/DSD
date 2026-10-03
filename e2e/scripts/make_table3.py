#!/usr/bin/env python
"""Collect <out>/<method>_N<n>/dsd_metrics.json into the paper's Table 3 layout.

Two blocks: dsdSim's metric convention (what the simulator reports, so rows
line up with sim runs) and the standard serving convention.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

NAMES = {"vanilla": "Vanilla SD", "specpp": "SpecDec++", "eagle3": "EAGLE3"}


def load(out: Path):
    runs = {}
    for p in out.glob("*_N*/dsd_metrics.json"):
        m = re.fullmatch(r"(.+)_N(\d+)", p.parent.name)
        if m and m.group(1) in NAMES:
            runs[(m.group(1), int(m.group(2)))] = json.loads(p.read_text())["system_metrics"]
    return runs


def fmt(x, nd=1):
    return "–" if x is None else f"{x:.{nd}f}"


def block(runs, title, rows):
    ns = sorted({n for _, n in runs})
    lines = [f"### {title}", "", "| Method | Metric | " + " | ".join(f"N={n}" for n in ns) + " |",
             "|---|---|" + "---|" * len(ns)]
    for method in NAMES:
        if not any(m == method for m, _ in runs):
            continue
        for i, (label, get, nd) in enumerate(rows):
            vals = [fmt(get(runs[(method, n)]), nd) if (method, n) in runs else "–" for n in ns]
            lines.append(f"| {NAMES[method] if i == 0 else ''} | {label} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main(out: str) -> int:
    runs = load(Path(out))
    if not runs:
        print(f"no runs under {out}")
        return 1
    sim = [("TTFT (ms)", lambda s: s["sim"]["ttft_ms"]["mean"], 1),
           ("TPOT (ms)", lambda s: s["sim"]["tpot_ms_pooled"], 1),
           ("Thr. (req/s)", lambda s: s["sim"]["throughput_rps"], 2),
           ("accept ratio", lambda s: s["sim"]["acceptance_ratio"], 3)]
    std = [("TTFT (ms)", lambda s: s["standard"]["ttft_ms"]["mean"], 1),
           ("TPOT (ms)", lambda s: s["standard"]["tpot_ms"]["mean"], 1),
           ("Thr. (req/s)", lambda s: s["standard"]["throughput_rps"], 2),
           ("mean accept len", lambda s: s["standard"]["mean_acceptance_length"], 2)]
    print(block(runs, "dsdSim convention", sim))
    print()
    print(block(runs, "Standard convention", std))
    bad = [k for k, s in runs.items() if not s.get("accounting_ok", True)]
    if bad:
        print(f"\n**accounting check failed for:** {sorted(bad)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "runs/table3"))
