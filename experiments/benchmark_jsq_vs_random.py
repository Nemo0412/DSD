#!/usr/bin/env python3
"""Benchmark: JSQ vs Random routing on a fat-tree datacenter topology.

Draft and target servers are randomly distributed across the fat-tree.
Compares TPOT (Time Per Output Token) and throughput between the two policies.

Usage (from dsdSim/src/):
    python ../experiments/benchmark_jsq_vs_random.py
"""
from __future__ import annotations

import json
import random
import sys
import os
from typing import Dict, List

# Allow running from any directory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from sim import load_config_from_mapping, simulate_config_obj
from network.fat_tree import FatTreeTopology


# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------

SIM_TIME_MS   = 8_000     # simulation window
LOAD_RPS      = 90.0      # arrival rate (requests / second) — high load to stress targets
N_TARGETS     = 4         # number of target servers
N_DRAFTS      = 12        # number of draft servers
GAMMA         = 3         # speculative tokens per round
ACCEPTANCE    = 0.70      # fixed token acceptance rate
SEED          = 2025      # master seed for GPU placement
ROUTERS       = ["jsq", "random"]

# Simulated GPU compute: realistic numbers for a 70B target on A100
# At these values: verify(3 tokens)≈1.5ms → ~50ms/conv → ~62% util per target
TARGET_PREFILL_MS_PER_TOKEN = 0.05   # 50 µs / token (prefill)
TARGET_DECODE_MS_PER_TOKEN  = 0.50   # 500 µs / token (decode / verify)
ANSWER_LENGTH_MEAN = 100
ANSWER_LENGTH_STD  = 30

# Fat-tree parameters (match fat_tree_config.json defaults)
FAT_TREE = {
    "gpus_per_node": 4,
    "nodes_per_pod":  4,
    "pods_per_root":  4,
}


# ---------------------------------------------------------------------------
# GPU placement
# ---------------------------------------------------------------------------

def sample_gpu_indices(n_drafts: int, n_targets: int, seed: int) -> tuple[list, list]:
    """Randomly draw unique GPU indices for drafts and targets.

    Placement guarantees:
    - All indices are distinct (one device per GPU).
    - Drawn uniformly from [0, total_gpus).
    """
    topo = FatTreeTopology(FAT_TREE)
    rng  = random.Random(seed)

    total = topo.total_gpus
    indices = rng.sample(range(total), n_drafts + n_targets)

    draft_indices  = indices[:n_drafts]
    target_indices = indices[n_drafts:]
    return draft_indices, target_indices


def describe_placement(draft_indices, target_indices) -> str:
    topo = FatTreeTopology(FAT_TREE)
    lines = ["  GPU placement:"]
    for i, gi in enumerate(draft_indices):
        pod, node, local = topo.gpu_location(gi)
        lines.append(f"    draft_{i:02d}  → GPU {gi:3d}  (pod {pod}, node {node}, local {local})")
    for i, gi in enumerate(target_indices):
        pod, node, local = topo.gpu_location(gi)
        lines.append(f"    target_{i:02d} → GPU {gi:3d}  (pod {pod}, node {node}, local {local})")
    # Latency matrix
    lines.append("  One-way draft→target latencies (ns):")
    header = "           " + "".join(f"  tgt{j}" for j in range(len(target_indices)))
    lines.append(header)
    for i, gi in enumerate(draft_indices):
        row = f"    draft_{i:02d}"
        for gj in target_indices:
            ns = topo.latency_ns(gi, gj)
            row += f"  {ns:5.0f}"
        lines.append(row)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def make_config(
    router: str,
    draft_indices: list,
    target_indices: list,
    sim_seed: int = 1,
) -> dict:
    """Build a complete simulation config dict for the given router."""
    return {
        "sim_time_ms": SIM_TIME_MS,
        "seed": sim_seed,
        "verbose": False,
        "debug": False,
        "router": router,
        "router_params": {},
        "global_router": "disabled",
        "gamma": GAMMA,
        "execution_mode": "blocking",

        "speculation": {
            "framework": "vanilla",
            "execution_mode": "distributed",
            "acceptance": {
                "disable_model": True,
                "default_rate": ACCEPTANCE,
            },
        },

        "performance_model": {"type": "default"},

        "network": {"enabled": True},

        "workload": {
            "arrival": "poisson",
            "rate_rps": LOAD_RPS,
        },
        "answer_length_mean": ANSWER_LENGTH_MEAN,
        "answer_length_std":  ANSWER_LENGTH_STD,
        "answer_length_min":  20,
        "answer_length_max":  300,
        "use_answer_distribution": True,

        "scheduler": {
            "type": "baseline",
            "pools": {
                "all_targets": {"clusters": ["dc_cluster"]},
            },
            "prefill": {
                "pool": "all_targets",
                "queue_policy": "fcfs",
                "max_batch_requests": 8,
                "max_wait_ms": 0.5,
                "delayed_batch_ms": 0.3,
                "max_queue_depth": 64,
                "backpressure_wait_ms": 0.05,
            },
            "decode": {
                "pool": "all_targets",
                "queue_policy": "fcfs",
                "max_batch_requests": 8,
                "max_wait_ms": 0.5,
                "delayed_batch_ms": 0.3,
                "max_queue_depth": 64,
                "backpressure_wait_ms": 0.05,
            },
        },

        "auto_topology": {
            "clusters": [
                {
                    "name": "dc_cluster",
                    "router": router,
                    "router_params": {},
                    "targets": {
                        "count": N_TARGETS,
                        "tiers": [
                            {
                                "name": "target_tier",
                                "count": N_TARGETS,
                                "model": "llama-70b",
                                "gpu": "A100",
                                "batch_window_ms": 0.8,
                                "batch_size": 8,
                                # Realistic per-token compute latencies so targets
                                # become the bottleneck under high load.
                                "metadata": {
                                    "prefill_latency_per_token": TARGET_PREFILL_MS_PER_TOKEN,
                                    "decode_latency_per_token":  TARGET_DECODE_MS_PER_TOKEN,
                                },
                            }
                        ],
                    },
                    "drafts": {
                        "count": N_DRAFTS,
                        "draft_bucket_labels": ["draft"],
                        "count_by_label": {"draft": N_DRAFTS},
                        "metadata_by_label": {
                            "draft": {
                                "hardware": "A100",
                                "model_name": "llama-7b",
                            }
                        },
                    },
                    "connectivity": {
                        "fanout_per_draft": N_TARGETS,   # each draft can route to any target
                        "network_model": {
                            "type": "fat_tree",
                            **FAT_TREE,
                            "latency": {
                                "nvlink_ns":      60,
                                "node_to_pod_ns": 150,
                                "pod_to_root_ns": 250,
                            },
                            # Pass explicit random GPU placements.
                            "draft_gpu_indices":  draft_indices,
                            "target_gpu_indices": target_indices,
                        },
                    },
                }
            ]
        },
    }


# ---------------------------------------------------------------------------
# Run one configuration and extract key metrics
# ---------------------------------------------------------------------------

def run_scenario(router: str, draft_indices: list, target_indices: list) -> dict:
    cfg = load_config_from_mapping(make_config(router, draft_indices, target_indices))
    mj = simulate_config_obj(cfg, emit_output=False)   # returns full metrics_json dict
    return {
        "router":                  router,
        "tpot_avg_ms":             mj.get("tpot_avg_ms", float("nan")),
        "tpot_p50_ms":             mj.get("tpot_p50_ms", float("nan")),
        "tpot_p95_ms":             mj.get("tpot_p95_ms", float("nan")),
        "throughput_rps":          mj.get("throughput_jobs_s", float("nan")),
        "goodput_rps":             mj.get("goodput_rps", float("nan")),
        "acceptance_rate":         mj.get("acceptance_rate", float("nan")),
        "rtt_avg_ms":              mj.get("rtt_avg_ms", float("nan")),
        "completed_conversations": mj.get("completed_conversation_count", 0),
        "effective_tok_s":         mj.get("effective_tok_s", float("nan")),
    }


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_results(results: list[dict], draft_indices, target_indices) -> None:
    topo = FatTreeTopology(FAT_TREE)

    # Latency statistics
    lats = []
    for gi in draft_indices:
        for gj in target_indices:
            lats.append(topo.latency_ns(gi, gj))
    unique_lats = sorted(set(lats))

    print("\n" + "=" * 68)
    print("  JSQ vs Random — Fat-tree Datacenter  (draft–target, no edge)")
    print("=" * 68)
    print(f"  Topology : {FAT_TREE['pods_per_root']} pods × "
          f"{FAT_TREE['nodes_per_pod']} nodes × "
          f"{FAT_TREE['gpus_per_node']} GPUs  "
          f"= {topo.total_gpus} total GPUs")
    print(f"  Devices  : {N_DRAFTS} drafts, {N_TARGETS} targets "
          f"(randomly distributed, seed={SEED})")
    print(f"  Load     : {LOAD_RPS} req/s × {SIM_TIME_MS/1000:.0f}s "
          f"≈ {int(LOAD_RPS*SIM_TIME_MS/1000)} requests")
    verify_ms = GAMMA * TARGET_DECODE_MS_PER_TOKEN
    rounds = ANSWER_LENGTH_MEAN / GAMMA
    conv_ms = rounds * verify_ms
    util_pct = LOAD_RPS * conv_ms / (N_TARGETS * 1000) * 100
    print(f"  Compute  : verify≈{verify_ms:.1f}ms/round, "
          f"~{rounds:.0f} rounds/conv → ~{conv_ms:.0f}ms compute/conv")
    print(f"  Target util (est): {util_pct:.0f}%  "
          f"({'overloaded' if util_pct > 85 else 'loaded' if util_pct > 50 else 'light'})")
    print(f"  Latencies: {unique_lats} ns  "
          f"(same-node=60, same-pod=300, cross-pod=800)")
    lat_counts = {lat: lats.count(lat) for lat in unique_lats}
    print(f"  Pair distribution: {lat_counts}")
    print()

    col = 16
    hdr = f"  {'Metric':<28} " + "".join(f"{r['router']:>{col}}" for r in results)
    print(hdr)
    print("  " + "-" * (28 + col * len(results) + 1))

    def row(label, key, fmt=".3f"):
        vals = [results[i].get(key, float("nan")) for i in range(len(results))]
        line = f"  {label:<28} " + "".join(f"{v:{col}{fmt}}" for v in vals)
        print(line)

    row("TPOT avg (ms)",           "tpot_avg_ms")
    row("TPOT p50 (ms)",           "tpot_p50_ms")
    row("TPOT p95 (ms)",           "tpot_p95_ms")
    row("Throughput (jobs/s)",     "throughput_rps",  ".1f")
    row("Goodput (conv/s)",        "goodput_rps",     ".1f")
    row("Effective tok/s",         "effective_tok_s", ".1f")
    row("Acceptance rate",         "acceptance_rate", ".3f")
    row("RTT avg (ms)",            "rtt_avg_ms")
    row("Completed convs",         "completed_conversations", ".0f")
    print()

    # Highlight winner
    tpots = [(r["tpot_avg_ms"], r["router"]) for r in results]
    best = min(tpots)[1]
    print(f"  ✓ Lower TPOT (better): {best.upper()}")
    thrpt = [(r["throughput_rps"], r["router"]) for r in results]
    best_t = max(thrpt)[1]
    print(f"  ✓ Higher throughput  : {best_t.upper()}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    draft_indices, target_indices = sample_gpu_indices(N_DRAFTS, N_TARGETS, SEED)

    print("\nFat-tree GPU placement:")
    print(describe_placement(draft_indices, target_indices))

    results = []
    for router in ROUTERS:
        print(f"\n[Running {router.upper()} ...]", flush=True)
        res = run_scenario(router, draft_indices, target_indices)
        results.append(res)
        print(f"  → TPOT avg={res['tpot_avg_ms']:.3f}ms  "
              f"throughput={res['throughput_rps']:.1f} jobs/s  "
              f"goodput={res['goodput_rps']:.1f} conv/s")

    print_results(results, draft_indices, target_indices)
