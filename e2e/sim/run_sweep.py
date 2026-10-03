#!/usr/bin/env python
"""Sweep methods x N through dsdSim and build the sim side of the small Table 3.

    export DSDSIM_ROOT=/path/to/dsdSim
    SIMPY=/path/to/simvenv/bin/python
    $SIMPY sim/run_sweep.py                                  # vanilla specpp eagle3, N=1 2 4 7
    $SIMPY sim/run_sweep.py --methods "vanilla specpp" --ns "1 2" --mode lockstep \
        --calibration sim/calibration.json --out sim/results/lockstep

For every (method, N) it writes <out>/configs/<method>_N<n>/{config.yaml,trace.jsonl,sim.log},
<out>/<method>_N<n>.json (table row + diagnostics + per-request records), and
finally <out>/table3_sim.md in the "dsdSim convention" layout of
scripts/make_table3.py.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_config  # noqa: E402

NAMES = {"vanilla": "Vanilla SD", "specpp": "SpecDec++", "eagle3": "EAGLE3 (sim `eagle`)",
         "eagle3_chain": "EAGLE3 (chain proxy)"}
# method -> make_config args. eagle3_chain = dsdSim's vanilla chain drafting
# (gamma 4) charged at the EAGLE3-head draft latency: the closest thing to our
# real EAGLE3 path the unmodified sim can express (see README).
SIM_ARGS = {"vanilla": ["--method", "vanilla"], "specpp": ["--method", "specpp"],
            "eagle3": ["--method", "eagle"],
            "eagle3_chain": ["--method", "vanilla", "--draft-profile", "eagle"]}


def fmt(x, nd=1):
    return "–" if x is None else f"{x:.{nd}f}"


def block(runs: dict, title: str, rows) -> str:
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


def table(out: Path) -> str:
    runs = {}
    for p in out.glob("*_N*.json"):
        stem, _, n = p.stem.rpartition("_N")
        if stem in NAMES and n.isdigit():
            runs[(stem, int(n))] = json.loads(p.read_text())
    if not runs:
        return f"no results under {out}"
    sim = [("TTFT (ms)", lambda r: r["table"]["ttft_mean_ms"], 1),
           ("TPOT (ms)", lambda r: r["table"]["tpot_pooled_ms"], 1),
           ("Thr. (req/s)", lambda r: r["table"]["throughput_rps"], 2),
           ("accept ratio", lambda r: r["table"]["acceptance_ratio"], 3)]
    diag = [("mean gamma", lambda r: r["diagnostics"]["mean_gamma_per_round"], 2),
            ("accepted/round (no bonus)", lambda r: r["diagnostics"]["mean_accepted_per_round_no_bonus"], 2),
            ("output/target tokens", lambda r: r["diagnostics"]["mean_output_over_target_tokens"], 2),
            ("target util", lambda r: r["diagnostics"]["target_utilization"], 2),
            ("jobs/target batch", lambda r: r["diagnostics"]["target_avg_batch_items"], 1),
            ("completed", lambda r: float(r["diagnostics"]["completed_conversations"]), 0)]
    any_r = next(iter(runs.values()))
    c = any_r["config"]
    head = (f"dsdSim, mode={c.get('mode')}, per_draft_rps={c.get('per_draft_rps')}, rtt={c.get('rtt_ms')} ms, "
            f"seed={c.get('seed')}, calibration={c.get('calibration')}")
    status = ""
    try:
        status = json.loads(Path(c["calibration"]).read_text()).get("_status", "")
    except Exception:
        pass
    parts = [f"<!-- {head} -->", f"_{head}_", ""]
    if status:
        parts += [f"**Calibration status:** {status}", ""]
    parts += [block(runs, "dsdSim convention", sim), "", block(runs, "Sim diagnostics", diag)]
    bad = [f"{m}_N{n}" for (m, n), r in sorted(runs.items()) if not r["diagnostics"]["all_completed"]]
    if bad:
        parts.append(f"\n**not all requests completed in:** {bad}")
    return "\n".join(parts)


def run_one(method: str, n: int, a) -> str:
    tag = f"{method}_N{n}"
    cdir = a.out / "configs" / tag
    res = a.out / f"{tag}.json"
    if res.exists() and not a.force:
        return f"skip {tag} (exists)"
    argv = SIM_ARGS[method] + ["--n-drafters", str(n), "--calibration", str(a.calibration),
            "--per-draft-rps", str(a.per_draft_rps), "--rtt-ms", str(a.rtt_ms), "--seed", str(a.seed),
            "--mode", a.mode, "--out-dir", str(cdir), "--dsdsim-root", str(a.dsdsim_root)]
    if a.n_requests:
        argv += ["--n-requests", str(a.n_requests)]
    if a.acceptance_profile:
        argv += ["--acceptance-profile", str(a.acceptance_profile)]
    if a.target_tokens_scale != 1.0:
        argv += ["--target-tokens-scale", str(a.target_tokens_scale)]
    argv += list(a.extra or [])
    cfg_path = make_config.write(make_config.parse(argv))
    env = dict(os.environ, DSDSIM_ROOT=str(a.dsdsim_root), PYTHONWARNINGS="ignore")
    p = subprocess.run([a.python, str(HERE / "run_sim.py"), str(cfg_path), "-o", str(res),
                        "--log", str(cdir / "sim.log")], env=env, capture_output=True, text=True)
    if p.returncode != 0:
        return f"FAIL {tag}\n{p.stderr[-3000:]}"
    return p.stdout.strip()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", default="vanilla specpp eagle3 eagle3_chain")
    ap.add_argument("--ns", default="1 2 4 7")
    ap.add_argument("--calibration", type=Path, default=HERE / "calibration.json")
    ap.add_argument("--per-draft-rps", type=float, default=1.0)
    ap.add_argument("--rtt-ms", type=float, default=10.0)
    ap.add_argument("--n-requests", type=int, default=None, help="default min(400, 120*N)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=("async", "lockstep"), default="async")
    ap.add_argument("--acceptance-profile", type=Path, default=None)
    ap.add_argument("--target-tokens-scale", type=float, default=1.0)
    ap.add_argument("--dsdsim-root", type=Path, default=os.environ.get("DSDSIM_ROOT"))
    ap.add_argument("--python", default=sys.executable, help="interpreter of the sim venv")
    ap.add_argument("--out", type=Path, default=HERE / "results")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--force", action="store_true", help="re-run even if the result JSON exists")
    ap.add_argument("--table-only", action="store_true")
    ap.add_argument("extra", nargs="*", help="extra make_config.py args after --")
    a = ap.parse_args(argv)
    if not a.dsdsim_root:
        ap.error("set DSDSIM_ROOT or pass --dsdsim-root")
    a.dsdsim_root = Path(a.dsdsim_root).resolve()
    a.out.mkdir(parents=True, exist_ok=True)
    if not a.table_only:
        todo = [(m, int(n)) for m in a.methods.split() for n in a.ns.split()]
        for m, _ in todo:
            if m not in NAMES:
                ap.error(f"unknown method {m} (choose from {list(NAMES)})")
        with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
            for line in ex.map(lambda mn: run_one(mn[0], mn[1], a), todo):
                print(line, flush=True)
    md = table(a.out)
    (a.out / "table3_sim.md").write_text(md + "\n")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
