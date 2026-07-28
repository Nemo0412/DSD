#!/usr/bin/env python3
"""Rerun DSD-Sim on perturbed acceptance_seq traces (discrete-event, not first-order map)."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from perturb_acceptance_seq import corpus_stats  # noqa: E402
from sim import simulate_config  # noqa: E402


def load_rows(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text and not text.startswith("#"):
                rows.append(json.loads(text))
    return rows


def run_one(base_cfg: dict, trace_path: Path, out_json: Path, gamma: int) -> Dict[str, float]:
    cfg = dict(base_cfg)
    cfg["trace_path"] = str(trace_path)
    cfg["verbose"] = False
    cfg["debug"] = False
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
        yaml.safe_dump(cfg, tmp)
        tmp_path = Path(tmp.name)
    try:
        metrics = simulate_config(str(tmp_path), emit_output=False)
    finally:
        tmp_path.unlink(missing_ok=True)

    stats = corpus_stats(load_rows(trace_path), gamma=gamma)
    result = {
        "mean_accepted_per_window": float(stats["mean_accepted_per_window"]),
        "accept_bit_rate": float(stats["accept_bit_rate"]),
        "mean_burst_len": float(stats["mean_burst_len"]),
        "tpot_avg_ms": float(metrics.get("tpot_avg_ms") or metrics.get("avg_tpot_ms") or 0.0),
        "throughput_rps": float(
            metrics.get("conversation_throughput_rps")
            or metrics.get("goodput_rps")
            or metrics.get("throughput_jobs_s")
            or 0.0
        ),
        "effective_tok_s": float(metrics.get("effective_tok_s") or 0.0),
        "ttft_avg_ms": float(metrics.get("ttft_avg_ms") or metrics.get("avg_ttft_ms") or 0.0),
        "completed": float(metrics.get("completed_conversation_count") or 0.0),
        "acceptance_rate": float(metrics.get("acceptance_rate") or 0.0),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as handle:
        json.dump({"trace": str(trace_path), "metrics": {k: metrics[k] for k in metrics if not isinstance(metrics[k], (dict, list))}, "summary": result}, handle, indent=2)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--traces-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gamma", type=int, default=4)
    ap.add_argument("--glob", default="*.jsonl")
    args = ap.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as handle:
        base_cfg = yaml.safe_load(handle)

    traces = sorted(Path(args.traces_dir).glob(args.glob))
    if not traces:
        raise SystemExit(f"No traces matched {args.traces_dir}/{args.glob}")

    summary_rows = []
    for trace in traces:
        out_json = Path(args.out_dir) / f"{trace.stem}.json"
        print(f"=== Running {trace.name} ===", flush=True)
        row = run_one(base_cfg, trace, out_json, gamma=args.gamma)
        row["name"] = trace.stem
        summary_rows.append(row)
        print(
            f"  Lbar={row['mean_accepted_per_window']:.3f} "
            f"TPOT={row['tpot_avg_ms']:.2f} Thr={row['throughput_rps']:.2f} "
            f"tok/s={row['effective_tok_s']:.1f} completed={row['completed']:.0f}",
            flush=True,
        )

    summary_path = Path(args.out_dir) / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_rows, handle, indent=2)
    print(f"Wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
