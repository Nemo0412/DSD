#!/usr/bin/env python3
"""Compare one real EAGLE-tree run with a 1-target / 1-draft simulator replay.

The real jsonl is the SafeAILab tree (whole tree verified each round). The
simulator replays that run's accepted-path lengths, charges the measured
draft / verify / comm, and sends total_tokens nodes to the target.
"""
from __future__ import annotations

import importlib.util
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIM_PY = Path("/scratch/ll5914/dsd-simvenv/bin/python")


def mean(xs):
    return statistics.fmean(xs) if xs else 0.0


def load_rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def real_request(row):
    rounds = row["tree_accept"]
    parts = []
    for step in rounds:
        parts.append(
            float(step["draft_generation_ms"])
            + float(step["target_verification_ms"])
            + float(step.get("comm_ms", 0.0))
        )
    e2e = sum(parts)
    ttft = parts[0] if parts else 0.0
    out = max(1, int(row["target_tokens"]))
    tpot = (e2e - ttft) / max(out - 1, 1)
    return {"ttft": ttft, "e2e": e2e, "tpot": tpot, "out": out, "rounds": rounds}


def fit(rows):
    draft, verify, comm, first_extra = [], [], [], []
    nodes, depth, accepted = [], [], []
    for row in rows:
        steps = row["tree_accept"]
        if not steps:
            continue
        steady = steps[1:] or steps
        d = mean([float(s["draft_generation_ms"]) for s in steady])
        v = mean([float(s["target_verification_ms"]) for s in steady])
        c = mean([float(s.get("comm_ms", 0.0)) for s in steady])
        draft.append(d)
        verify.append(v)
        comm.append(c)
        first = steps[0]
        first_cost = (
            float(first["draft_generation_ms"])
            + float(first["target_verification_ms"])
            + float(first.get("comm_ms", 0.0))
        )
        first_extra.append(first_cost - (d + v + c))
        for s in steady:
            nodes.append(int(s.get("tree_nodes", 60)))
            depth.append(int(s["depth"]))
            accepted.append(int(s["accepted_path_len"]))
    return {
        "draft": mean(draft),
        "verify": mean(verify),
        "comm": mean(comm),
        "prefill": max(0.0, mean(first_extra)),
        "tree_nodes": int(round(mean(nodes) or 60)),
        "max_depth": max(depth or [7]),
        "mean_accept": mean(accepted),
    }


def write_inputs(dataset_dir: Path, rows):
    dataset_dir.mkdir(parents=True, exist_ok=True)
    trace = []
    step_lines = []
    for row in rows:
        rid = row["request_id"]
        trace.append({
            "request_id": rid,
            "arrival_ms": 0.0,
            "prompt_tokens": int(row["prompt_tokens"]),
            "target_tokens": max(1, int(row["target_tokens"])),
            "device_tier": "default",
        })
        for step in row["tree_accept"]:
            step_lines.append(json.dumps({
                "request_id": rid,
                "i": int(step["round"]),
                "gamma": int(step["depth"]),
                "accepted": int(step["accepted_path_len"]),
            }))
    (dataset_dir / "trace_src.jsonl").write_text("".join(json.dumps(r) + "\n" for r in trace))
    (dataset_dir / "steps.jsonl").write_text("\n".join(step_lines) + "\n")


def run_one(dataset: str, src: Path, out_root: Path, stats: dict):
    rows = load_rows(src)
    real = [real_request(r) for r in rows]
    work = out_root / dataset
    write_inputs(work, rows)
    calib = {
        "_status": "EAGLE tree, 1 L40S target + 1 L40S head, replay of this run",
        "_units": "milliseconds",
        "target": {
            "prefill_base_ms": stats["prefill"],
            "prefill_ms_per_token": 0.0,
            "verify_base_ms": stats["verify"],
            "verify_ms_per_seq": 0.0,
            "verify_ms_per_token": 0.0,
            "verify_ms_per_ctx_token": 0.0,
            "mixed_batch_combine": "sum",
        },
        "draft": {
            "vanilla": {"base_ms": stats["draft"], "ms_per_token": 0.0, "ms_per_token_per_seq": 0.0},
            "specpp": {"base_ms": stats["draft"], "ms_per_token": 0.0, "ms_per_token_per_seq": 0.0},
            "eagle": {"base_ms": stats["draft"], "ms_per_token": 0.0, "ms_per_token_per_seq": 0.0},
        },
        "scheduling": {"target_batch_window_ms": 0.01, "scheduler_max_wait_ms": 0.0},
    }
    (work / "calibration.json").write_text(json.dumps(calib, indent=2))
    spec = importlib.util.spec_from_file_location("make_config", ROOT / "sim/make_config.py")
    mc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mc)
    od = work / "sim"
    cfg_path = mc.write(mc.parse([
        "--method", "eagle",
        "--n-drafters", "1",
        "--calibration", str(work / "calibration.json"),
        "--per-draft-rps", "0.02",
        "--rtt-ms", f"{stats['comm']:.6f}",
        "--n-requests", str(len(rows)),
        "--seed", "0",
        "--mode", "lockstep",
        "--gamma", "4",
        "--replay-from", str(work),
        "--max-num-seqs", "1",
        "--trace", str(work / "trace_src.jsonl"),
        "--dspec-sim-root", str(ROOT),
        "--out-dir", str(od),
        "--tail-ms", "600000",
        "--eagle-beam-width", "1",
        "--eagle-max-depth", str(max(stats["max_depth"], 1)),
        "--eagle-total-tokens", str(stats["tree_nodes"]),
        "--draft-profile", "eagle",
    ]))
    # Replay sets target_tokens = rounds * gamma. Put back the real output length
    # so the bonus-aware eagle loop stops on the measured token count.
    by_id = {r["request_id"]: max(1, int(r["target_tokens"])) for r in rows}
    trace_path = od / "trace.jsonl"
    fixed = []
    for line in trace_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        rec["target_tokens"] = by_id[rec["request_id"]]
        fixed.append(rec)
    trace_path.write_text("".join(json.dumps(r) + "\n" for r in fixed))
    import yaml
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["_dspec_real"]["comm_ms"] = stats["comm"]
    cfg["_dspec_real"]["tree_nodes"] = stats["tree_nodes"]
    cfg["_dspec_real"]["model_pair"] = "Qwen2-7B + EAGLE head, 2x L40S"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, width=120))
    result_path = od / "result.json"
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["DSPEC_SIM_ROOT"] = str(ROOT)
    subprocess.run(
        [str(SIM_PY), str(ROOT / "sim/run_sim.py"), str(cfg_path),
         "-o", str(result_path), "--log", str(od / "sim.log"),
         "--dspec-sim-root", str(ROOT)],
        check=True,
        env=env,
    )
    sim = json.loads(result_path.read_text())
    sim_by = {r["request_id"]: r for r in sim["requests"]}
    return {
        "n": len(rows),
        "real_ttft": mean([r["ttft"] for r in real]),
        "real_tpot": mean([r["tpot"] for r in real]),
        "real_e2e": mean([r["e2e"] for r in real]),
        "sim_ttft": sim["table"]["ttft_mean_ms"],
        "sim_tpot": sim["table"]["tpot_pooled_ms"],
        "sim_e2e": mean([sim_by[r["request_id"]]["total_latency_ms"] for r in rows]),
        "completed": f"{sim['diagnostics']['completed_conversations']}/{sim['diagnostics']['n_requests_in_trace']}",
        "nodes": stats["tree_nodes"],
        "accept": stats["mean_accept"],
        "draft": stats["draft"],
        "verify": stats["verify"],
        "comm": stats["comm"],
    }


def main() -> int:
    real_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "")
    if not real_dir.is_dir():
        raise SystemExit(f"usage: {sys.argv[0]} REAL_DIR")
    out = real_dir / "sim_align"
    print(f"{'dataset':12} {'n':>3} {'real_ttft':>10} {'sim_ttft':>10} {'real_tpot':>10} {'sim_tpot':>10} {'real_e2e':>10} {'sim_e2e':>10} {'e2e_err':>8} nodes accept draft verify comm")
    for name in ("gsm8k", "cnndm", "humaneval"):
        src = real_dir / f"{name}.jsonl"
        if not src.exists():
            continue
        rows = load_rows(src)
        stats = fit(rows)
        (out / name).mkdir(parents=True, exist_ok=True)
        (out / name / "fit.json").write_text(json.dumps(stats, indent=2))
        row = run_one(name, src, out, stats)
        err = (row["sim_e2e"] - row["real_e2e"]) / row["real_e2e"] if row["real_e2e"] else 0.0
        print(
            f"{name:12} {row['n']:3d} {row['real_ttft']:10.1f} {row['sim_ttft']:10.1f} "
            f"{row['real_tpot']:10.1f} {row['sim_tpot']:10.1f} {row['real_e2e']:10.1f} {row['sim_e2e']:10.1f} "
            f"{err:8.1%} {row['nodes']:5d} {row['accept']:6.2f} {row['draft']:6.2f} {row['verify']:6.2f} {row['comm']:6.3f} "
            f"done {row['completed']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
