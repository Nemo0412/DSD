#!/usr/bin/env python
"""Real vs simulated Table 3, side by side, in dsdSim's metric convention.

    python scripts/compare_real_sim.py runs/table3 sim/results/measured > runs/table3/real_vs_sim.md

Real rows come from <real>/<method>_N<n>/dsd_metrics.json ("sim" block of
system_metrics, i.e. the collector's dsdSim-convention numbers); sim rows from
<sim>/<method>_N<n>.json ("table"). EAGLE3 is compared against the sim's
`eagle3_chain` proxy (plain chain drafting at the head's cost): the sim's own
EAGLE model ends a request at its first zero-acceptance round
(sim/README.md), which makes it meaningless here.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

METHODS = {"vanilla": ("Vanilla SD", "vanilla"), "specpp": ("SpecDec++", "specpp"),
           "eagle3": ("EAGLE3", "eagle3_chain")}
METRICS = [("TTFT (ms)", lambda r: r["ttft_ms"]["mean"], "ttft_mean_ms"),
           ("TPOT (ms)", lambda r: r["tpot_ms_pooled"], "tpot_pooled_ms"),
           ("Thr. (req/s)", lambda r: r["throughput_rps"], "throughput_rps")]


def main(real_dir: str, sim_dir: str) -> int:
    real, sim = {}, {}
    for p in Path(real_dir).glob("*_N*/dsd_metrics.json"):
        m = re.fullmatch(r"(.+)_N(\d+)", p.parent.name)
        if m and m.group(1) in METHODS:
            real[(m.group(1), int(m.group(2)))] = json.loads(p.read_text())["system_metrics"]["sim"]
    for p in Path(sim_dir).glob("*_N*.json"):
        m = re.fullmatch(r"(.+)_N(\d+)", p.stem)
        if m:
            sim[(m.group(1), int(m.group(2)))] = json.loads(p.read_text())["table"]
    ns = sorted({n for _, n in real})
    print("| Method | Metric | " + " | ".join(f"N={n} real / sim (err)" for n in ns) + " |")
    print("|---|---|" + "---|" * len(ns))
    errs = []
    for key, (name, sim_key) in METHODS.items():
        if not any(k == key for k, _ in real):
            continue
        for i, (label, get_real, sim_field) in enumerate(METRICS):
            cells = []
            for n in ns:
                r = real.get((key, n))
                s = sim.get((sim_key, n))
                if r is None or s is None:
                    cells.append("–")
                    continue
                rv, sv = get_real(r), s[sim_field]
                err = (sv - rv) / rv if rv else float("nan")
                errs.append(abs(err))
                cells.append(f"{rv:.1f} / {sv:.1f} ({err:+.0%})")
            print(f"| {name if i == 0 else ''} | {label} | " + " | ".join(cells) + " |")
    if errs:
        errs.sort()
        print(f"\nMAPE over {len(errs)} cells: {sum(errs) / len(errs):.1%} "
              f"(median {errs[len(errs) // 2]:.1%}, max {errs[-1]:.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:3]))
