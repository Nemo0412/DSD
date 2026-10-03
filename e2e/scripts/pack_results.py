#!/usr/bin/env python
"""Pack a final_compare.sh output dir (runs/ is gitignored) into a small,
tracked results/<name>/: tables, deployment breakdown, calibration, profiles,
per-run system metrics and sim table cells.

    python scripts/pack_results.py runs/table3_client_final results/table3_client
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import sys


def main(src: str, dst: str) -> int:
    os.makedirs(dst, exist_ok=True)
    for f in ("table3.md", "deployment.md", "real_vs_sim_pooled.md", "real_vs_sim_tiered.md", "real_vs_sim_replay.md",
              "calibration.pooled.json", "calibration.eagle.json", "calibration.pooled.fit.json",
              "calibration.eagle.fit.json", "eagle3_profile.json"):
        if os.path.exists(f"{src}/{f}"):
            shutil.copy(f"{src}/{f}", f"{dst}/{f}")
    if os.path.exists(f"{src}/calib_g8/profile.json"):
        shutil.copy(f"{src}/calib_g8/profile.json", f"{dst}/specpp_acceptance_profile.json")
    keys = ("gamma", "gamma_policy", "draft_kind", "rtt_ms", "per_draft_rps", "n_requests", "n_drafters",
            "rate_rps", "chat_template", "no_prefix_caching", "client_side", "dtype")
    real = {}
    for p in sorted(glob.glob(f"{src}/*_N*/dsd_metrics.json")):
        run = os.path.dirname(p)
        d = json.load(open(p))
        entry = {"config": {k: d["config"].get(k) for k in keys}, "system_metrics": d["system_metrics"]}
        for extra in ("placement.json",):
            if os.path.exists(f"{run}/{extra}"):
                entry["placement"] = json.load(open(f"{run}/{extra}"))
        if os.path.exists(f"{run}/gpu_check.txt"):
            entry["gpu_check"] = open(f"{run}/gpu_check.txt").read().strip().split("\n")[-1]
        real[os.path.basename(run)] = entry
    json.dump(real, open(f"{dst}/real_system_metrics.json", "w"), indent=1)
    sim = {}
    for variant in ("sim_pooled", "sim_tiered", "sim_replay"):
        for p in sorted(glob.glob(f"{src}/{variant}/*_N*.json")):
            d = json.load(open(p))
            sim.setdefault(variant, {})[os.path.basename(p)[:-5]] = {
                **d["table"], "client_side_ttft_offset_ms": d["diagnostics"].get("client_side_ttft_offset_ms", 0.0)}
    json.dump(sim, open(f"{dst}/sim_tables.json", "w"), indent=1)
    print(f"packed {len(real)} real runs, {sum(len(v) for v in sim.values())} sim cells -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:3]))
