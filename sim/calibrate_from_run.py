#!/usr/bin/env python
"""Fit sim/calibration.json latency constants from a real dsd-real run.

Reads <run>/steps.jsonl (one row per sequence per propose() call; rows of
the same call share t_send_ms) and fits:

- target verify step: from a call's t_recv (drafts back) to the next call's
  t_send (the verify forward + bookkeeping of the step those drafts went
  into), against that step's batch: verify = base + per_seq * B +
  per_token * sum(gamma_i + 1). Steps that also carried a prefill (a new
  sequence appears in the next call) are excluded: their time is dominated
  by the prompt, which the prefill terms model separately.
- target prefill: steps that also prefilled new sequences, their time over
  the verify model's prediction regressed on the new prompts' total tokens
  (combined additively with the verify part, as in one vLLM step).
- draft round: the drafter's own draft_ms per call (it serves the whole
  call's batch in one generate()), against gamma and batch size:
  draft = base + per_token * max_gamma + per_token_per_seq * max_gamma * (B - 1).

    python sim/calibrate_from_run.py runs/table3/vanilla_N1 --method vanilla \\
        --base sim/calibration.json -o sim/calibration.measured.json

Prints fit quality (R^2, residual spread); a poor fit is a reason to look at
the data before trusting the simulator's output.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_calls(run: Path):
    rows = [json.loads(line) for line in (run / "steps.jsonl").read_text().split("\n") if line.strip()]
    calls = defaultdict(list)
    for r in rows:
        calls[round(r["t_send_ms"], 3)].append(r)
    keys = sorted(calls)
    return [calls[k] for k in keys]


def fit(X: np.ndarray, y: np.ndarray):
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum()) or 1e-9
    return coef, 1 - ss_res / ss_tot, float(np.percentile(np.abs(y - pred), 90))


def _collect(calls, vX, vy, dX, dy, pre=None, prompt_len=None) -> None:
    seqs_seen: set[str] = set()
    for i, call in enumerate(calls):
        seqs_seen |= {r["request_id"] for r in call}
        gammas = [r["n_draft"] for r in call]
        B = len(call)
        g = max(gammas)
        dX.append([1.0, g, g * (B - 1)])
        dy.append(call[0]["draft_ms"])
        if i + 1 < len(calls):
            nxt = calls[i + 1]
            verify = nxt[0]["t_send_ms"] - call[0]["t_recv_ms"]
            fresh = {r["request_id"] for r in nxt} - seqs_seen
            if fresh:
                # the step also prefilled new sequences: kept aside for the
                # prefill fit (residual over the verify model)
                if pre is not None and prompt_len is not None:
                    pre.append((B, sum(x + 1 for x in gammas),
                                sum(prompt_len.get(r, 0) for r in fresh), verify))
                continue
            vX.append([1.0, B, sum(x + 1 for x in gammas)])
            vy.append(verify)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="+", help="one or more run dirs, pooled into one fit")
    ap.add_argument("--method", choices=("vanilla", "specpp", "eagle"), required=True,
                    help="which draft block of the calibration to fill")
    ap.add_argument("--base", default=str(Path(__file__).with_name("calibration.json")))
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    vX, vy, dX, dy, pre = [], [], [], [], []
    n_calls = 0
    for run in args.run:
        calls = load_calls(Path(run))
        n_calls += len(calls)
        per = json.loads((Path(run) / "dsd_metrics.json").read_text())["per_request"]
        _collect(calls, vX, vy, dX, dy, pre, {r["request_id"]: r["prompt_len"] for r in per})

    cal = json.loads(Path(args.base).read_text())
    report = {"runs": args.run, "calls": n_calls}

    if len(vy) >= 10:
        c, r2, p90 = fit(np.array(vX), np.array(vy))
        cal["target"].update(verify_base_ms=round(float(c[0]), 3),
                             verify_ms_per_seq=round(float(c[1]), 4),
                             verify_ms_per_token=round(float(c[2]), 4))
        report["verify"] = {"n": len(vy), "base": c[0], "per_seq": c[1], "per_token": c[2],
                            "r2": r2, "abs_resid_p90_ms": p90, "mean_ms": float(np.mean(vy))}
    if len(vy) >= 10 and len(pre) >= 10:
        b0, bs, bt = (cal["target"][k] for k in ("verify_base_ms", "verify_ms_per_seq",
                                                  "verify_ms_per_token"))
        P = np.array(pre)
        extra = P[:, 3] - (b0 + bs * P[:, 0] + bt * P[:, 1])
        c, r2, p90 = fit(np.column_stack([np.ones(len(P)), P[:, 2]]), extra)
        # vLLM runs a prefill inside the same step as the verifies (additive)
        cal["target"].update(prefill_base_ms=round(float(c[0]), 3),
                             prefill_ms_per_token=round(float(c[1]), 4),
                             mixed_batch_combine="sum")
        report["prefill"] = {"n": len(P), "base": c[0], "per_prompt_token": c[1], "r2": r2,
                             "abs_resid_p90_ms": p90}
    if len(dy) >= 10:
        D = np.array(dX)
        if np.ptp(D[:, 1]) == 0:
            # gamma never varied: an intercept and a per-token slope are the
            # same column. Attribute it all to per-token (base = 0), which is
            # what the sim's draft model scales with gamma.
            c2, r2, p90 = fit(D[:, 1:], np.array(dy))
            c = np.array([0.0, *c2])
            report["draft_note"] = "gamma constant in this run: base fixed at 0"
        else:
            c, r2, p90 = fit(D, np.array(dy))
        cal["draft"][args.method].update(base_ms=round(float(c[0]), 3),
                                         ms_per_token=round(float(c[1]), 4),
                                         ms_per_token_per_seq=round(float(c[2]), 4))
        report["draft"] = {"n": len(dy), "base": c[0], "per_token": c[1], "per_token_per_seq": c[2],
                           "r2": r2, "abs_resid_p90_ms": p90, "mean_ms": float(np.mean(dy))}
    cal["_status"] = f"MEASURED from {args.run} (target verify + draft.{args.method}); other fields as in {args.base}"
    Path(args.out).write_text(json.dumps(cal, indent=2))
    print(json.dumps(report, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
