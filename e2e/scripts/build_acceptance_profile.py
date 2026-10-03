#!/usr/bin/env python
"""Per-position conditional acceptance from a calibration run, for SpecDec++.

    python scripts/build_acceptance_profile.py runs/calib_g8 -o runs/calib_g8/profile.json \\
        --edges 64,96

Calibrate with a static gamma at least as large as SpecDec++'s max (8):
positions beyond the calibration gamma are never observed, only padded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dsdreal.policy.window import AcceptanceProfile


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--edges", default="", help="context-length bucket edges, e.g. 64,96")
    ap.add_argument("--min-count", type=int, default=20)
    args = ap.parse_args()
    run = Path(args.run_dir)
    metrics = json.loads((run / "dsd_metrics.json").read_text())
    ctx = {r["request_id"]: r["prompt_len"] for r in metrics["per_request"]}
    rows = [json.loads(line) for line in (run / "steps.jsonl").read_text().split("\n") if line.strip()]
    edges = [int(e) for e in args.edges.split(",") if e]
    prof = AcceptanceProfile.from_steps(rows, ctx, edges=edges, min_count=args.min_count)
    prof.to_json(args.out)
    print(json.dumps({"rounds": len(rows), "edges": prof.edges,
                      "probs": [[round(p, 3) for p in row] for row in prof.probs]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
