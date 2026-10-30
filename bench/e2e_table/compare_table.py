#!/usr/bin/env python3
"""Real e2e vs the existing simulator rows for the 16-GPU and 10-GPU tables.

Simulator numbers are read from the result.json files already under results/.
Real numbers come from a campaign directory written by run_campaign.sh:
  <real>/<pair>/<dataset>/{dsd_metrics.json,steps.jsonl}
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path("/home/ll5914/DSD-MLsys2027")
DATASETS = ("gsm8k", "cnndm", "humaneval")
LABELS = {
    "qwen15_7b": "Qwen2.5-1.5B / Qwen2.5-7B",
    "qwen7_14b": "Qwen2.5-7B / Qwen2.5-14B",
    "eagle_qwen2_7b": "Qwen2-7B + EAGLE head",
    "tinyllama_llama2_7b": "TinyLlama-1.1B / Llama-2-7B",
    "llama31_eagle3": "Llama-3.1-8B + EAGLE3 head",
}
# Left-hand table in the request is labeled 4xA100/12xL40S. The Qwen rows
# were simulated with A100 target latency. The Llama rows that match the
# pasted numbers were simulated with H100 target latency (4xH100/12xL40S).
SIM = {
    "a100x4_l40sx12": {
        "qwen15_7b": ROOT / "results/sim_a100x4_l40sx12/pairs/qwen15_7b",
        "qwen7_14b": ROOT / "results/sim_a100x4_l40sx12/pairs/qwen7_14b",
        "eagle_qwen2_7b": ROOT / "results/sim_a100x4_l40sx12/eagle_measured",
        "tinyllama_llama2_7b": ROOT / "results/sim_h100x4_l40sx12/llama32_1b_31_8b",
        "llama31_eagle3": ROOT / "results/sim_h100x4_l40sx12/llama31_eagle3",
    },
    # 1xA100 + 3xL40S. No simulator row for this size yet.
    "a100x1_l40sx3": {
        "qwen15_7b": ROOT / "results/sim_a100x1_l40sx3/qwen15_7b",
        "qwen7_14b": ROOT / "results/sim_a100x1_l40sx3/qwen7_14b",
        "eagle_qwen2_7b": ROOT / "results/sim_a100x1_l40sx3/eagle_qwen2_7b",
        "tinyllama_llama2_7b": ROOT / "results/sim_a100x1_l40sx3/tinyllama_llama2_7b",
        "llama31_eagle3": ROOT / "results/sim_a100x1_l40sx3/llama31_eagle3",
    },
    # Real run is 2xA100 + 8xL40S. Simulator rows are the 2-target/8-draft H100 table.
    "a100x2_l40sx8": {
        "qwen15_7b": ROOT / "results/sim_h100x2_l40sx8/qwen15_7b",
        "qwen7_14b": ROOT / "results/sim_h100x2_l40sx8/qwen7_14b",
        "eagle_qwen2_7b": ROOT / "results/sim_h100x2_l40sx8/eagle_qwen2_7b",
        "tinyllama_llama2_7b": ROOT / "results/sim_h100x2_l40sx8/llama32_1b_31_8b",
        "llama31_eagle3": ROOT / "results/sim_h100x2_l40sx8/llama31_eagle3",
    },
    "h100x2_l40sx8": {
        "qwen15_7b": ROOT / "results/sim_h100x2_l40sx8/qwen15_7b",
        "qwen7_14b": ROOT / "results/sim_h100x2_l40sx8/qwen7_14b",
        "eagle_qwen2_7b": ROOT / "results/sim_h100x2_l40sx8/eagle_qwen2_7b",
        "tinyllama_llama2_7b": ROOT / "results/sim_h100x2_l40sx8/llama32_1b_31_8b",
        "llama31_eagle3": ROOT / "results/sim_h100x2_l40sx8/llama31_eagle3",
    },
}
METRICS = (
    "Comm (ms)", "TTFT (ms)", "TPOT (ms)", "Throughput (req/s)", "E2E (ms)",
    "Accept len", "Accept ratio",
)


def _mean(xs: list[float]) -> float | None:
    return statistics.fmean(xs) if xs else None


def sim_row(path: Path) -> dict[str, float] | None:
    f = path / "result.json"
    if not f.is_file():
        return None
    doc = json.loads(f.read_text())
    table = doc["table"]
    # mean_accepted_per_round_no_bonus excludes the bonus token. Adding 1
    # matches the real run's mean_acceptance_length, which includes it.
    accepted = doc.get("diagnostics", {}).get("mean_accepted_per_round_no_bonus")
    return {
        "Comm (ms)": float(doc["config"].get("comm_ms", doc["config"].get("rtt_ms", 0.0))),
        "TTFT (ms)": float(table["ttft_mean_ms"]),
        "TPOT (ms)": float(table["tpot_pooled_ms"]),
        "Throughput (req/s)": float(table["throughput_rps"]),
        "E2E (ms)": float(doc["sim_metrics_json"]["avg_conversation_ms"]),
        "Accept len": None if accepted is None else float(accepted) + 1.0,
        "Accept ratio": float(table["acceptance_ratio"]),
    }


def real_row(path: Path) -> dict[str, float] | None:
    metrics_path = path / "dsd_metrics.json"
    steps_path = path / "steps.jsonl"
    if not metrics_path.is_file():
        return None
    doc = json.loads(metrics_path.read_text())
    sim = doc["system_metrics"].get("sim") or {}
    if not sim or sim.get("ttft_ms") is None:
        return None
    e2e = [r["e2e_ms"] for r in doc["per_request"] if r.get("e2e_ms") is not None]
    comm = []
    if steps_path.is_file():
        for line in steps_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("net_excess_ms") is not None:
                comm.append(float(row["net_excess_ms"]))
    ttft = sim["ttft_ms"]["mean"]
    standard = doc["system_metrics"].get("standard") or {}
    return {
        "Comm (ms)": _mean(comm) if comm else float("nan"),
        "TTFT (ms)": float(ttft),
        "TPOT (ms)": float(sim["tpot_ms_pooled"]),
        "Throughput (req/s)": float(sim["throughput_rps"]),
        "E2E (ms)": float(_mean(e2e) if e2e else float("nan")),
        "Accept len": float(standard["mean_acceptance_length"]) if standard.get("mean_acceptance_length") is not None else None,
        "Accept ratio": float(sim["acceptance_ratio"]) if sim.get("acceptance_ratio") is not None else None,
    }


def fmt(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:.2f}" if abs(v) < 20 else f"{v:.1f}"


def render(topology: str, real_root: Path | None) -> str:
    pairs = SIM[topology]
    lines = [f"## {topology}", ""]
    header = "| Pair | Metric | " + " | ".join(
        f"{ds} real | {ds} sim" for ds in DATASETS
    ) + " |"
    sep = "|" + "---|" * (2 + 2 * len(DATASETS))
    lines += [header, sep]
    for key, sim_dir in pairs.items():
        for i, metric in enumerate(METRICS):
            cells = [LABELS[key] if i == 0 else "", metric]
            for ds in DATASETS:
                real = real_row(real_root / key / ds) if real_root else None
                sim = sim_row(sim_dir / ds)
                cells.append(fmt(None if real is None else real[metric]))
                cells.append(fmt(None if sim is None else sim[metric]))
            lines.append("| " + " | ".join(cells) + " |")
    if topology == "a100x4_l40sx12":
        lines.append(
            "Qwen rows use the 4xA100/12xL40S simulator results. "
            "TinyLlama-1.1B / Llama-2-7B replaces gated Llama-3.1-8B. "
            "Its sim column is still Llama-3.2-1B / Llama-3.1-8B on 4xH100/12xL40S. "
            "Llama-3.1-8B + EAGLE3 has no open substitute: the head is trained on Llama-3.1 hidden states."
        )
        lines.append("")
    lines.append(
        "TinyLlama-1.1B / Llama-2-7B is the runnable pair: same tokenizer, "
        "and Llama-3.2-1B cannot draft for Llama-2 (vocab 128256 vs 32000). "
        "The sim column is still Llama-3.2-1B / Llama-3.1-8B. "
        "EAGLE3-LLaMA has no open substitute."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", type=Path, default=None, help="campaign output dir")
    ap.add_argument("--topology", default="a100x4_l40sx12")
    args = ap.parse_args()
    if args.topology == "all":
        text = "\n".join(render(t, None) for t in SIM)
    else:
        text = render(args.topology, args.real)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
