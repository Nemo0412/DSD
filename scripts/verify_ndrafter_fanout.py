#!/usr/bin/env python
"""N-drafter fan-out gate: one target, RoundRobin across >=2 real drafters
over real ZMQ, all serving the identical draft model.

Two things must both hold:
1. Losslessness carries over from Stage 2a/2b: fp32 + greedy is deterministic
   and draft proposals are just token ids on the wire, so which of the N
   (identical) drafters served a given request must not change the output --
   exact match against a non-spec fp32 baseline, same as verify_fused_real_draft.py.
2. RoutingPolicy actually fans out: with N prompts and N drafters, RoundRobin
   must spread fresh-request assignments roughly evenly, not pile them onto
   one drafter (dsdreal/policy/routing.py; recorded via
   dsdreal.target.proposer.configure(..., assignments=sink), not by scraping
   per-drafter server logs).

Run against >=2 draft servers already listening, one --draft-addr per
drafter (see run_draft_server.py --drafter-id):

    python scripts/verify_ndrafter_fanout.py \\
        --draft-addr d0=tcp://127.0.0.1:5560 --draft-addr d1=tcp://127.0.0.1:5561
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

TARGET_MODEL_DEFAULT = os.environ.get("DSD_TARGET_MODEL", "Qwen/Qwen3-1.7B")
GAMMA = 4
MAX_LEN = 1024
MAX_TOKENS = 32
BALANCE_TOL = 0.34  # RR over few prompts/drafters: allow some rounding slack

PROMPTS = [
    "The capital of France is",
    "In one sentence, explain why the sky is blue:",
    "List three prime numbers:",
    "def fibonacci(n):",
    "Translate 'good morning' to Spanish:",
    "The opposite of hot is",
    "Name a primary color:",
    "2 + 2 =",
]


def _quiet_env() -> None:
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _cleanup_gpu() -> None:
    from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

    cleanup_dist_env_and_memory()


def greedy(llm, prompts, max_tokens):
    from vllm import SamplingParams

    outs = llm.generate(
        prompts, SamplingParams(temperature=0.0, max_tokens=max_tokens), use_tqdm=False
    )
    return [list(o.outputs[0].token_ids) for o in outs]


def parse_draft_addr(pairs: list[str]) -> dict[str, str]:
    addresses = {}
    for pair in pairs:
        drafter_id, addr = pair.split("=", 1)
        addresses[drafter_id] = addr
    if len(addresses) < 2:
        raise ValueError(f"need >=2 --draft-addr entries for a fan-out test, got {addresses}")
    return addresses


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft-addr", action="append", required=True,
                     help="drafter_id=tcp://host:port, repeat once per drafter")
    ap.add_argument("--target-model", default=TARGET_MODEL_DEFAULT)
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    args = ap.parse_args()
    addresses = parse_draft_addr(args.draft_addr)

    _quiet_env()
    from vllm import LLM

    from dsdreal.policy.routing import RoundRobin
    from dsdreal.proto.transport import ZmqDraftClient
    from dsdreal.target import proposer

    common = dict(enforce_eager=True, dtype="float32", disable_log_stats=True,
                  max_model_len=MAX_LEN)

    # 1. fp32 baseline, speculation off.
    baseline_llm = LLM(model=args.target_model, gpu_memory_utilization=args.gpu_frac, **common)
    baseline = greedy(baseline_llm, PROMPTS, MAX_TOKENS)
    del baseline_llm
    _cleanup_gpu()

    # 2. Speculative, routed round-robin across every drafter in `addresses`.
    client = ZmqDraftClient(addresses)
    assignments: list[str] = []
    proposer.configure(client, routing=RoundRobin(), assignments=assignments)

    target = LLM(
        model=args.target_model, gpu_memory_utilization=args.gpu_frac,
        speculative_config={
            "method": "custom_class",
            "model": "dsdreal.target.proposer.RemoteDraftProposer",
            "num_speculative_tokens": GAMMA,
        },
        **common,
    )
    fanned_out = greedy(target, PROMPTS, MAX_TOKENS)
    del target
    _cleanup_gpu()

    proposer.configure(None)
    client.close()

    exact_match = fanned_out == baseline
    if not exact_match:
        for i, (b, r) in enumerate(zip(baseline, fanned_out)):
            if b != r:
                div = next((j for j, (x, y) in enumerate(zip(b, r)) if x != y),
                           min(len(b), len(r)))
                print(f"  prompt[{i}] diverged at token {div}: baseline={b} fanned_out={r}")

    counts = Counter(assignments)
    n_drafters = len(addresses)
    expected_share = 1.0 / n_drafters
    shares = {d: counts.get(d, 0) / len(assignments) for d in addresses} if assignments else {}
    balanced = bool(assignments) and all(
        abs(share - expected_share) <= BALANCE_TOL for share in shares.values()
    ) and set(counts) == set(addresses)

    result = {
        "n_drafters": n_drafters,
        "n_requests_routed": len(assignments),
        "assignment_counts": dict(counts),
        "assignment_shares": {k: round(v, 4) for k, v in shares.items()},
        "exact_token_match": exact_match,
        "balanced": balanced,
    }
    print(json.dumps(result, indent=2))

    ok = exact_match and balanced
    if not ok:
        print(f"exact_token_match={exact_match} balanced={balanced}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
