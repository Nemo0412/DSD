#!/usr/bin/env python3
"""Benchmark: FIFO vs LAB (Length-Aware Batching) on fat-tree datacenter.

LAB groups requests by context-length bucket so each batch contains jobs with
similar sequence lengths, reducing padding waste and average batch compute time.

Routing is fixed at JSQ so we isolate the batching policy effect.

Usage (from anywhere):
    python experiments/benchmark_batching.py
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from sim import load_config_from_mapping, simulate_config_obj
from network.fat_tree import FatTreeTopology

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------

SIM_TIME_MS  = 10_000   # longer window for more stable stats
LOAD_RPS     = 90.0     # overloaded regime so batching choices matter
N_TARGETS    = 4
N_DRAFTS     = 12
GAMMA        = 3
ACCEPTANCE   = 0.70
SEED_GPU     = 2025     # GPU placement seed (same as routing benchmark)
SEED_SIM     = 1        # SimPy RNG seed

# Realistic GPU compute — same as routing benchmark
TARGET_PREFILL_MS = 0.05   # ms / token
TARGET_DECODE_MS  = 0.50   # ms / token

# Answer lengths: wide distribution to produce heterogeneous context lengths
ANSWER_LENGTH_MEAN = 200
ANSWER_LENGTH_STD  = 80
ANSWER_LENGTH_MIN  = 30
ANSWER_LENGTH_MAX  = 600

FAT_TREE = {"gpus_per_node": 4, "nodes_per_pod": 4, "pods_per_root": 4}

# Batching policies to compare.
# Each entry: (label, queue_policy, lab_bucket_size)
POLICIES = [
    ("FIFO",    "fifo",     0),
    ("LAB-64",  "lab",     64),   # 64-token buckets
    ("LAB-128", "lab",    128),   # coarser bucketing
]


# ---------------------------------------------------------------------------
# GPU placement (reuse same helper as routing benchmark)
# ---------------------------------------------------------------------------

def sample_gpu_indices(n_drafts: int, n_targets: int, seed: int):
    import random
    topo = FatTreeTopology(FAT_TREE)
    rng  = random.Random(seed)
    indices = rng.sample(range(topo.total_gpus), n_drafts + n_targets)
    return indices[:n_drafts], indices[n_drafts:]


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def make_config(
    queue_policy: str,
    lab_bucket_size: int,
    draft_indices: list,
    target_indices: list,
) -> dict:
    sched_phase = {
        "pool": "all_targets",
        "queue_policy": queue_policy,
        "lab_bucket_size": lab_bucket_size,
        "max_batch_requests": 8,
        "max_wait_ms": 0.5,
        "delayed_batch_ms": 0.3,
        "max_queue_depth": 128,
        "backpressure_wait_ms": 0.05,
    }
    return {
        "sim_time_ms": SIM_TIME_MS,
        "seed": SEED_SIM,
        "verbose": False,
        "debug": False,
        "router": "jsq",          # fixed routing — only batching varies
        "router_params": {},
        "global_router": "disabled",
        "gamma": GAMMA,
        "execution_mode": "blocking",

        "speculation": {
            "framework": "vanilla",
            "execution_mode": "distributed",
            "acceptance": {"disable_model": True, "default_rate": ACCEPTANCE},
        },

        "performance_model": {"type": "default"},
        "network": {"enabled": True},

        "workload": {"arrival": "poisson", "rate_rps": LOAD_RPS},
        "answer_length_mean": ANSWER_LENGTH_MEAN,
        "answer_length_std":  ANSWER_LENGTH_STD,
        "answer_length_min":  ANSWER_LENGTH_MIN,
        "answer_length_max":  ANSWER_LENGTH_MAX,
        "use_answer_distribution": True,

        "scheduler": {
            "type": "baseline",
            "pools": {"all_targets": {"clusters": ["dc_cluster"]}},
            "prefill": sched_phase,
            "decode":  sched_phase,
        },

        "auto_topology": {
            "clusters": [{
                "name": "dc_cluster",
                "router": "jsq",
                "targets": {
                    "count": N_TARGETS,
                    "tiers": [{
                        "name": "target_tier",
                        "count": N_TARGETS,
                        "model": "llama-70b",
                        "gpu": "A100",
                        "batch_window_ms": 0.8,
                        "batch_size": 8,
                        "metadata": {
                            "prefill_latency_per_token": TARGET_PREFILL_MS,
                            "decode_latency_per_token":  TARGET_DECODE_MS,
                        },
                    }],
                },
                "drafts": {
                    "count": N_DRAFTS,
                    "draft_bucket_labels": ["draft"],
                    "count_by_label": {"draft": N_DRAFTS},
                    "metadata_by_label": {
                        "draft": {"hardware": "A100", "model_name": "llama-7b"},
                    },
                },
                "connectivity": {
                    "fanout_per_draft": N_TARGETS,
                    "network_model": {
                        "type": "fat_tree",
                        **FAT_TREE,
                        "latency": {
                            "nvlink_ns": 60,
                            "node_to_pod_ns": 150,
                            "pod_to_root_ns": 250,
                        },
                        "draft_gpu_indices":  draft_indices,
                        "target_gpu_indices": target_indices,
                    },
                },
            }],
        },
    }


# ---------------------------------------------------------------------------
# Run one policy and extract metrics
# ---------------------------------------------------------------------------

def run_scenario(label: str, queue_policy: str, lab_bucket_size: int,
                 draft_indices: list, target_indices: list) -> dict:
    cfg = load_config_from_mapping(
        make_config(queue_policy, lab_bucket_size, draft_indices, target_indices)
    )
    mj = simulate_config_obj(cfg, emit_output=False)
    return {
        "label":           label,
        "tpot_avg_ms":     mj.get("tpot_avg_ms",    float("nan")),
        "tpot_p50_ms":     mj.get("tpot_p50_ms",    float("nan")),
        "tpot_p95_ms":     mj.get("tpot_p95_ms",    float("nan")),
        "throughput_rps":  mj.get("throughput_jobs_s", float("nan")),
        "goodput_rps":     mj.get("goodput_rps",     float("nan")),
        "effective_tok_s": mj.get("effective_tok_s", float("nan")),
        "rtt_avg_ms":      mj.get("rtt_avg_ms",      float("nan")),
        "completed_convs": mj.get("completed_conversation_count", 0),
        "ttft_avg_ms":     mj.get("ttft_avg_ms",     float("nan")),
    }


# ---------------------------------------------------------------------------
# Pretty results table
# ---------------------------------------------------------------------------

def print_results(results: list[dict], draft_indices: list, target_indices: list) -> None:
    topo = FatTreeTopology(FAT_TREE)
    lats = [topo.latency_ns(gi, gj) for gi in draft_indices for gj in target_indices]
    lat_dist = {lat: lats.count(lat) for lat in sorted(set(lats))}

    print("\n" + "=" * 72)
    print("  FIFO vs LAB — Batching Policy Comparison on Fat-tree Datacenter")
    print("=" * 72)
    print(f"  Topology   : {FAT_TREE['pods_per_root']} pods × "
          f"{FAT_TREE['nodes_per_pod']} nodes × "
          f"{FAT_TREE['gpus_per_node']} GPUs = {topo.total_gpus} total GPUs")
    print(f"  Devices    : {N_DRAFTS} drafts, {N_TARGETS} targets  (seed={SEED_GPU})")
    print(f"  Load       : {LOAD_RPS} req/s × {SIM_TIME_MS/1000:.0f}s  "
          f"(routing fixed at JSQ)")
    verify_ms = GAMMA * TARGET_DECODE_MS
    rounds = ANSWER_LENGTH_MEAN / GAMMA
    print(f"  Compute    : verify≈{verify_ms:.1f}ms/round, "
          f"answer μ={ANSWER_LENGTH_MEAN}±{ANSWER_LENGTH_STD} tokens "
          f"[{ANSWER_LENGTH_MIN},{ANSWER_LENGTH_MAX}]")
    print(f"  Latencies  : {lat_dist} ns")
    print()

    col = 14
    labels = [r["label"] for r in results]
    hdr = f"  {'Metric':<30} " + "".join(f"{lb:>{col}}" for lb in labels)
    print(hdr)
    print("  " + "-" * (30 + col * len(results) + 1))

    def row(name, key, fmt=".3f"):
        vals = [r.get(key, float("nan")) for r in results]
        print(f"  {name:<30} " + "".join(f"{v:{col}{fmt}}" for v in vals))

    row("TPOT avg (ms)",           "tpot_avg_ms")
    row("TPOT p50 (ms)",           "tpot_p50_ms")
    row("TPOT p95 (ms)",           "tpot_p95_ms")
    row("TTFT avg (ms)",           "ttft_avg_ms")
    row("RTT avg (ms)",            "rtt_avg_ms")
    row("Throughput (jobs/s)",     "throughput_rps",  ".1f")
    row("Goodput (conv/s)",        "goodput_rps",     ".1f")
    row("Effective tok/s",         "effective_tok_s", ".1f")
    row("Completed convs",         "completed_convs", ".0f")
    print()

    best_tpot = min(results, key=lambda r: r["tpot_avg_ms"])["label"]
    best_tput = max(results, key=lambda r: r["throughput_rps"])["label"]
    print(f"  ✓ Lower TPOT (better)    : {best_tpot}")
    print(f"  ✓ Higher throughput      : {best_tput}")

    # improvement vs FIFO baseline
    fifo = next((r for r in results if r["label"] == "FIFO"), None)
    if fifo:
        print()
        print("  Improvement vs FIFO baseline:")
        for r in results:
            if r["label"] == "FIFO":
                continue
            d_tpot = (fifo["tpot_avg_ms"] - r["tpot_avg_ms"]) / fifo["tpot_avg_ms"] * 100
            d_tput = (r["throughput_rps"] - fifo["throughput_rps"]) / fifo["throughput_rps"] * 100
            print(f"    {r['label']:<10} TPOT {d_tpot:+.1f}%   Throughput {d_tput:+.1f}%")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    draft_indices, target_indices = sample_gpu_indices(N_DRAFTS, N_TARGETS, SEED_GPU)

    results = []
    for label, policy, bucket in POLICIES:
        print(f"\n[Running {label} (queue_policy={policy}, bucket={bucket}) ...]",
              flush=True)
        res = run_scenario(label, policy, bucket, draft_indices, target_indices)
        results.append(res)
        print(f"  → TPOT avg={res['tpot_avg_ms']:.3f}ms  "
              f"throughput={res['throughput_rps']:.1f} jobs/s  "
              f"goodput={res['goodput_rps']:.1f} conv/s")

    print_results(results, draft_indices, target_indices)
