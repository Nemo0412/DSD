#!/usr/bin/env python
"""Acceptance-length parity gate: does routing draft calls over real ZMQ
change how many tokens get accepted per round, versus the identical draft
model called in-process (no transport) on the same node?

Stage 2a (verify_fused_real_draft.py) proved token-for-token losslessness
against a non-spec baseline. Stage 2b (verify_distributed_rtt_sweep.py)
proved TPOT rises linearly with injected RTT, but its L_bar_effective is
inferred backwards from a timing slope -- it never measures acceptance
length directly. This script does: RemoteDraftProposer.configure(...,
stats=sink) (dsdreal/target/proposer.py) appends one int per non-fresh
propose() round -- the number of tokens confirmed since the prior round for
that slot, i.e. accepted draft tokens + the bonus token. That is the
ground-truth per-round acceptance length, straight from vLLM's own
num_tokens_no_spec bookkeeping, no rejection-sampler monkeypatch needed.

Same target+draft models, same prompts, same gamma, fp32 + enforce_eager +
greedy (deterministic) run twice: once through InProcessDraftClient (Stage
2a's fused path, no network), once through ZmqDraftClient against a real
draft server over loopback ZMQ (Stage 2b's path). Since draft proposals are
just token ids serialized over the wire, transport should not change *what*
gets accepted, only *how long it takes* -- so both legs should produce the
same output tokens and statistically indistinguishable acceptance-length
distributions.

Run against a draft server already listening (see run_draft_server.py):

    python scripts/verify_acceptance_length_parity.py \\
        --draft-addr tcp://127.0.0.1:5560
"""
from __future__ import annotations

import argparse
import json
import os
import statistics

TARGET_MODEL_DEFAULT = os.environ.get("DSD_TARGET_MODEL", "Qwen/Qwen3-1.7B")
DRAFT_MODEL_DEFAULT = os.environ.get("DSD_DRAFT_MODEL", "Qwen/Qwen3-0.6B")
GAMMA = 4
MAX_LEN = 1024
MAX_TOKENS = 128
REL_TOL = 0.03

PROMPTS = [
    "The capital of France is",
    "In one sentence, explain why the sky is blue:",
    "List three prime numbers:",
    "def fibonacci(n):",
]


def _quiet_env() -> None:
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
    # RemoteDraftProposer.configure() sets a module-level global; the
    # target's own engine-core must run in this process to see it.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _cleanup_gpu() -> None:
    # See verify_fused_real_draft.py: LLMEngine.__del__ never tears down
    # engine_core, so a plain `del` leaves CUDA memory/NCCL groups reserved.
    from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

    cleanup_dist_env_and_memory()


def greedy(llm, prompts, max_tokens):
    from vllm import SamplingParams

    outs = llm.generate(
        prompts, SamplingParams(temperature=0.0, max_tokens=max_tokens), use_tqdm=False
    )
    return [list(o.outputs[0].token_ids) for o in outs]


def _spec_target(model, gamma, gpu_frac):
    from vllm import LLM

    return LLM(
        model=model, gpu_memory_utilization=gpu_frac, enforce_eager=True,
        dtype="float32", disable_log_stats=True, max_model_len=MAX_LEN,
        speculative_config={
            "method": "custom_class",
            "model": "dsdreal.target.proposer.RemoteDraftProposer",
            "num_speculative_tokens": gamma,
        },
    )


def run_fused(target_model, draft_model, target_gpu_frac, draft_gpu_frac):
    from vllm import LLM

    from dsdreal.draft.engine_vllm import VllmDraftEngine
    from dsdreal.proto.transport import InProcessDraftClient
    from dsdreal.target import proposer

    draft_llm = LLM(model=draft_model, gpu_memory_utilization=draft_gpu_frac,
                     enforce_eager=True, dtype="float32", disable_log_stats=True,
                     max_model_len=MAX_LEN)
    draft_engine = VllmDraftEngine(draft_llm, node_id="d0")
    stats: list[int] = []
    proposer.configure(InProcessDraftClient(draft_engine, node_id="t0"), stats=stats)

    target = _spec_target(target_model, GAMMA, target_gpu_frac)
    tokens = greedy(target, PROMPTS, MAX_TOKENS)
    del target
    _cleanup_gpu()

    proposer.configure(None)
    del draft_llm, draft_engine
    _cleanup_gpu()
    return tokens, stats


def run_distributed(target_model, draft_addr, target_gpu_frac):
    from dsdreal.proto.transport import ZmqDraftClient
    from dsdreal.target import proposer

    client = ZmqDraftClient({"d0": draft_addr})
    stats: list[int] = []
    proposer.configure(client, stats=stats)

    target = _spec_target(target_model, GAMMA, target_gpu_frac)
    tokens = greedy(target, PROMPTS, MAX_TOKENS)
    del target
    _cleanup_gpu()

    proposer.configure(None)
    client.close()
    return tokens, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft-addr", required=True)
    ap.add_argument("--target-model", default=TARGET_MODEL_DEFAULT)
    ap.add_argument("--draft-model", default=DRAFT_MODEL_DEFAULT)
    ap.add_argument("--target-gpu-frac", type=float, default=0.45)
    ap.add_argument("--draft-gpu-frac", type=float, default=0.20)
    ap.add_argument("--distributed-gpu-frac", type=float, default=0.85)
    args = ap.parse_args()

    _quiet_env()

    tokens_fused, stats_fused = run_fused(
        args.target_model, args.draft_model, args.target_gpu_frac, args.draft_gpu_frac)
    print(json.dumps({"leg": "fused", "n_rounds": len(stats_fused),
                       "mean_len": round(statistics.mean(stats_fused), 4)}))

    tokens_dist, stats_dist = run_distributed(
        args.target_model, args.draft_addr, args.distributed_gpu_frac)
    print(json.dumps({"leg": "distributed", "n_rounds": len(stats_dist),
                       "mean_len": round(statistics.mean(stats_dist), 4)}))

    exact_match = tokens_fused == tokens_dist
    if not exact_match:
        for i, (f, d) in enumerate(zip(tokens_fused, tokens_dist)):
            if f != d:
                div = next((j for j, (x, y) in enumerate(zip(f, d)) if x != y),
                           min(len(f), len(d)))
                print(f"  prompt[{i}] diverged at token {div}: fused={f} dist={d}")

    mean_fused = statistics.mean(stats_fused)
    mean_dist = statistics.mean(stats_dist)
    rel_diff = abs(mean_dist - mean_fused) / mean_fused
    round_count_diff = abs(len(stats_dist) - len(stats_fused))

    result = {
        "fused": {"n_rounds": len(stats_fused), "mean": round(mean_fused, 4),
                   "stdev": round(statistics.pstdev(stats_fused), 4)},
        "distributed": {"n_rounds": len(stats_dist), "mean": round(mean_dist, 4),
                         "stdev": round(statistics.pstdev(stats_dist), 4)},
        "rel_diff_of_means": round(rel_diff, 5),
        "round_count_diff": round_count_diff,
        "exact_token_match": exact_match,
    }
    print(json.dumps(result, indent=2))

    near_equal = rel_diff <= REL_TOL
    stable_round_count = round_count_diff <= 2
    ok = near_equal and stable_round_count
    if not ok:
        print(f"near_equal={near_equal} (rel_diff={rel_diff:.4f}, tol={REL_TOL}) "
              f"stable_round_count={stable_round_count} (diff={round_count_diff})")
    if not exact_match:
        print("note: acceptance-length gate passed but final tokens diverged -- "
              "investigate before trusting this result.")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
